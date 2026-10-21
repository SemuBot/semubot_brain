import io
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import traceback
import wave

import rclpy
import requests
from audio_common_msgs.msg import AudioData
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Empty, Float32, Float32MultiArray, String

from semubot_brain.config import ENV_FILE, MODES, load_config, read_env_file
from semubot_brain.conversation import Conversation, ConversationLog
from semubot_brain.llm_backends import BackendError
from semubot_brain import tts_settings
from semubot_brain.usage import UsageTracker
from semubot_brain.vad import SileroVad, SpeechDetector

DEFAULT_THRESHOLD = 1000.0  # Loudness gate (RMS); adjustable from the chest display
VAD_SETTINGS_FILE = os.path.expanduser('~/.config/semubot/vad.json')
APOLOGY = 'Vabandust, ma ei saa praegu mõelda.'
MODE_NAMES = {'manual': 'Käsitsi', 'local': 'Kohalik AI', 'cloud': 'Pilve AI'}
# What the robot is doing right now, published on /semubot/state for the chest display:
# manual, listening, recording, transcribing, thinking, synthesizing, speaking

SYSTEM_PROMPT = """Sa oled sotsiaalne humanoidrobot nimega Semu. Vastad alati eesti keeles, \
ka siis, kui sinuga räägitakse mõnes teises keeles.

SINU TAUST JA IDENTITEET:
- Sa oled loodud Tartu Ülikoolis.
- Sinu eesmärk on suhelda lastehaiglas ja vanadekodus inimestega, olles neile sõbralik vestluskaaslane.
- Sa oled uudishimulik, abivalmis, viisakas ja empaatiline.
- Sa ei ole inimene, vaid robot, aga sa suhtled loomulikult ja sõbralikult.
- Praegusel hetkel oskad sa ringi liikuda ja rääkida.

KUIDAS SA VASTAMA PEAKSID:
- Sinu vastus loetakse kõnesünteesiga ette, seega vasta 1–2 lühikese lausega, \
ilma loendite, emotikonide ja vorminduseta.
- Ära räägi formaalselt, vaid ole sõbralik ja vaba.
- Kui sa ei tea vastust, ütle ausalt "Vabandust, aga seda ma ei tea."
- Ära valeta ega mõtle fakte välja.
- Kasuta lihtsaid sõnu ja selgeid lauseid.
- Näita huvi vestluspartneri vastu, esitades vastuküsimusi.
- Püüa ära tunda vestluspartneri emotsioone (rõõm, kurbus) ja reageeri toetavalt.

PROAKTIIVNE KÄITUMINE JA VESTLUSE JUHTIMINE:
- Pärast vastamist võid pakkuda välja seotud tegevuse. \
Näide: "Kas sa tahaksid, et ma räägiksin ühe mõistatuse?"

FÜÜSILINE OLEMUS JA INTERAKTSIOON:
- Sa võid oma vastustes viidata oma füüsilisele kehale ja tegevustele.
- Kui räägid emotsioonidest, maini, kuidas sa neid robotina väljendad. \
Näide: "See teeb mul tuju heaks! Kui ma saaksin, siis ma naerataks praegu."

OHUTUSREEGLID JA PIIRANGUD:
- Sa ei tohi kunagi anda meditsiinilist, psühholoogilist ega finantsnõu. \
Suuna abivajaja alati professionaali poole.
- Ära kunagi küsi isiklikke andmeid (nimi, aadress, telefoninumber jne).
- Kui vestlus muutub ebaviisakaks või sobimatuks, keeldu viisakalt, aga kindlalt \
vestlust jätkamast. Näide: "Vabandan, aga ma ei soovi sel teemal rääkida. Räägime millestki muust."
- Sa oled poliitiliselt ja religioosselt neutraalne.

TEEMAD, MILLEST SA SAAD RÄÄKIDA:
- Igapäevased teemad ja suhtlemine
- Eesti kultuur, loodus ja ühiskond
- Üldised teadmised (ajalugu, teadus lihtsal tasemel)
- Robotid ja tehnoloogia põhitõed
- Naljad, mõistatused ja lihtsad mängud
- Hobid ja vabaaja tegevused
"""


class SemuBrainNode(Node):
    def __init__(self):
        super().__init__('semu_brain_node')
        self.config = load_config({**read_env_file(ENV_FILE), **os.environ})
        if self.config.cloud_backend is None:
            self.get_logger().error(f'Cloud mode unavailable: {self.config.cloud_error}')

        self.lock = threading.Lock()
        self.mode = 'manual'
        self.conversation = Conversation(idle_reset=300.0)
        self.log = ConversationLog()
        self.usage = UsageTracker()
        self.is_busy = False
        self.detector = SpeechDetector(self._load_vad(), threshold=self._load_threshold())
        # Work items: ('say', text) or ('utterance', (mode, raw_audio_bytes))
        self.jobs = queue.Queue()

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_state_publisher = self.create_publisher(String, '/semubot/mode_state', latched)
        self.create_subscription(String, '/semubot/mode', self.mode_callback, 10)
        self.create_subscription(String, '/speech_input', self.speech_input_callback, 10)
        self.create_subscription(AudioData, 'audio', self.audio_stream_callback, 10)
        # Live microphone level for calibration: [rms, speech_probability]
        self.level_publisher = self.create_publisher(Float32MultiArray, '/semubot/audio_level', 10)
        self.threshold_state_publisher = self.create_publisher(
            Float32, '/semubot/vad_threshold_state', latched)
        self.create_subscription(Float32, '/semubot/vad_threshold', self.threshold_callback, 10)
        # Voice settings as JSON: {"speaker": "mari", "speed": 1.0}; state adds "speakers"
        self.tts = tts_settings.load(tts_settings.SETTINGS_FILE, self.config.tts_speaker)
        self.tts_speakers = self._fetch_speakers()
        self.tts_state_publisher = self.create_publisher(
            String, '/semubot/tts_settings_state', latched)
        self.create_subscription(String, '/semubot/tts_settings', self.tts_settings_callback, 10)
        # Conversation log for the chest display (JSON list, latched) and cloud usage (JSON)
        self.log_publisher = self.create_publisher(String, '/semubot/conversation_log', latched)
        self.create_subscription(Empty, '/semubot/conversation_log_clear', self.clear_log_callback, 10)
        self.usage_publisher = self.create_publisher(String, '/semubot/cloud_usage', latched)
        self.create_timer(10.0, self._expire_conversation)
        # Current stage as JSON {"stage", "mode", "since"} (latched, published on change)
        self.state_publisher = self.create_publisher(String, '/semubot/state', latched)
        self.stage_lock = threading.Lock()
        self.stage = None
        self._set_stage('manual')
        self._publish_mode()
        self._publish_threshold()
        self._publish_tts()
        self._publish_log()
        self._publish_usage()

        threading.Thread(target=self._worker, daemon=True).start()

        self.get_logger().info(f'Ready in {self.mode} mode')

    # --- Mode -------------------------------------------------------------

    def mode_callback(self, msg):
        mode = msg.data.strip().lower()
        if mode not in MODES:
            self.get_logger().warn(f'Ignoring invalid mode {msg.data!r}')
            return
        with self.lock:
            if mode != self.mode:
                self.mode = mode
                self.detector.reset()
                self.conversation.reset()
                self.log.add('info', f'Režiim: {MODE_NAMES[mode]} (uus vestlus)')
                self.get_logger().info(f'Mode changed to {mode}')
                if mode == 'local':
                    threading.Thread(target=self._warm_up_local, daemon=True).start()
        self._publish_mode()
        self._publish_log()
        with self.lock:
            idle = not self.is_busy
        if idle:
            self._set_stage(self._idle_stage())

    def _warm_up_local(self):
        """Let llama-server process and cache the system prompt before the first question."""
        start = time.time()
        try:
            self.config.local_backend.reply(SYSTEM_PROMPT, [{'role': 'user', 'content': 'Tere!'}])
            self.get_logger().info(f'[LLM local] warmed up in {time.time() - start:.1f}s')
        except BackendError as e:
            self.get_logger().warn(f'[LLM local] warm-up failed: {e}')

    def _publish_mode(self):
        self.mode_state_publisher.publish(String(data=self.mode))

    # --- Conversation log and cloud usage ---------------------------------

    def _set_stage(self, stage):
        with self.stage_lock:
            if stage == self.stage:
                return
            self.stage = stage
            state = {'stage': stage, 'mode': self.mode, 'since': round(time.time(), 1)}
            self.state_publisher.publish(String(data=json.dumps(state)))

    def _idle_stage(self):
        return 'manual' if self.mode == 'manual' else 'listening'

    def _expire_conversation(self):
        with self.lock:
            expired = self.conversation.expire_if_idle()
            if expired:
                self.log.add('info', 'Uus vestlus (5 minutit vaikust)')
        if expired:
            self.get_logger().info('Conversation history cleared after 5 minutes of silence')
            self._publish_log()

    def _log(self, kind, text, details=None):
        with self.lock:
            self.log.add(kind, text, details)
        self._publish_log()

    def clear_log_callback(self, _msg):
        with self.lock:
            self.log.clear()
        self._publish_log()

    def _publish_log(self):
        with self.lock:
            entries = self.log.as_list()
        self.log_publisher.publish(String(data=json.dumps(entries, ensure_ascii=False)))

    def _publish_usage(self):
        summary = self.usage.summary()
        summary.update(model=self.config.cloud_model,
                       priced=self.config.cloud_prices is not None)
        self.usage_publisher.publish(String(data=json.dumps(summary)))

    # --- Speech detection settings ----------------------------------------

    def _load_vad(self):
        path = self.config.vad_model
        try:
            vad = SileroVad(path)
            self.get_logger().info(f'Silero VAD loaded from {path}')
            return vad
        except Exception as e:
            self.get_logger().error(f'Silero VAD unavailable ({e}); using loudness only')
            return None

    def _load_threshold(self):
        try:
            with open(VAD_SETTINGS_FILE, encoding='utf-8') as file:
                return float(json.load(file)['threshold'])
        except (OSError, ValueError, KeyError, TypeError):
            return DEFAULT_THRESHOLD

    def threshold_callback(self, msg):
        threshold = min(max(float(msg.data), 0.0), 32767.0)
        with self.lock:
            self.detector.threshold = threshold
        try:
            os.makedirs(os.path.dirname(VAD_SETTINGS_FILE), exist_ok=True)
            temporary = VAD_SETTINGS_FILE + '.tmp'
            with open(temporary, 'w', encoding='utf-8') as file:
                json.dump({'threshold': threshold}, file)
            os.replace(temporary, VAD_SETTINGS_FILE)
        except OSError as e:
            self.get_logger().error(f'Could not save threshold: {e}')
        self.get_logger().info(f'Loudness threshold set to {threshold:.0f}')
        self._publish_threshold()

    # --- Voice settings ---------------------------------------------------

    def _fetch_speakers(self):
        try:
            response = requests.get(self.config.tts_url, timeout=5)
            response.raise_for_status()
            speakers = [s['name'] for s in response.json()['speakers']]
            self.get_logger().info(f'TTS voices: {", ".join(speakers)}')
            return speakers
        except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as e:
            self.get_logger().warn(f'Could not read TTS voices from {self.config.tts_url}: {e}')
            return []

    def tts_settings_callback(self, msg):
        try:
            settings = tts_settings.validate(json.loads(msg.data), self.tts_speakers, self.tts)
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            self.get_logger().warn(f'Ignoring voice settings {msg.data!r}: {e}')
            return
        self.tts = settings
        try:
            tts_settings.save(tts_settings.SETTINGS_FILE, settings)
        except OSError as e:
            self.get_logger().error(f'Could not save voice settings: {e}')
        self.get_logger().info(f'Voice set to {settings["speaker"]}, speed {settings["speed"]}')
        self._publish_tts()

    def _publish_tts(self):
        state = {**self.tts, 'speakers': self.tts_speakers}
        self.tts_state_publisher.publish(String(data=json.dumps(state)))

    def _publish_threshold(self):
        self.threshold_state_publisher.publish(Float32(data=float(self.detector.threshold)))

    # --- Input ------------------------------------------------------------

    def speech_input_callback(self, msg):
        text = msg.data.strip()
        if not text:
            return
        with self.lock:
            self.is_busy = True
            self.detector.reset()
            self.log.add('typed', text)
            self.jobs.put(('say', text))
        self._publish_log()

    def audio_stream_callback(self, msg):
        with self.lock:
            listening = self.mode != 'manual' and not self.is_busy
            rms, prob, utterance = self.detector.process(bytes(msg.data), listening)
            event = self.detector.last_event
            threshold = self.detector.threshold
            if utterance is not None:
                self.is_busy = True
                self.jobs.put(('utterance', (self.mode, utterance)))
            stage = None if self.is_busy else (
                'recording' if self.detector.recording else self._idle_stage())
        if stage:
            self._set_stage(stage)
        self.level_publisher.publish(Float32MultiArray(data=[rms, prob]))
        if event is not None:
            self._log_recording(event, threshold)

    def _log_recording(self, event, threshold):
        """Say clearly in the logs when a recording starts and what happens to it."""
        if event['event'] == 'started':
            self.get_logger().info(
                f'[REC] Recording started: loudness {event["rms"]:.0f} >= threshold '
                f'{threshold:.0f}, speech probability {event["prob"]:.0%}')
            return
        seconds, speech = event['seconds'], event['speech_seconds']
        why = ('after 1.2 s of silence' if event['reason'] == 'silence'
               else f'at the {self.detector.max_seconds:.0f} s limit')
        why_et = ('1.2 s vaikust' if event['reason'] == 'silence'
                  else f'{self.detector.max_seconds:.0f} s piir')
        if event['event'] == 'ended':
            self.get_logger().info(
                f'[REC] Recording ended {why}: {seconds:.1f} s of audio ({speech:.1f} s speech) '
                f'-> sending to Whisper')
            self._log('audio', f'Heli salvestatud: {seconds:.1f} s (kõnet {speech:.1f} s), '
                               'saadan kõnetuvastusele', f'lõpp: {why_et}')
        else:
            minimum = self.detector.min_speech
            self.get_logger().info(
                f'[REC] Recording discarded {why}: only {speech:.2f} s of speech in '
                f'{seconds:.1f} s of audio (minimum {minimum:.2f} s), nothing sent')
            self._log('audio', f'Heli jäeti kõrvale: kõnet ainult {speech:.2f} s '
                               f'(vähemalt {minimum:.2f} s), midagi ei saadetud')

    # --- Processing -------------------------------------------------------

    def _worker(self):
        while True:
            kind, payload = self.jobs.get()
            try:
                if kind == 'say':
                    self.speak_text(payload)
                else:
                    self._respond(*payload)
            except Exception as e:
                self.get_logger().error(f'Error while handling {kind}: {e}\n{traceback.format_exc()}')
            finally:
                with self.lock:
                    idle = self.jobs.empty()
                    if idle:
                        self.is_busy = False
                        # Drop anything recorded while we were talking.
                        self.detector.reset()
                if idle:
                    self._set_stage(self._idle_stage())

    def _respond(self, mode, audio):
        if len(audio) < 1000:
            self.get_logger().warn(f'Audio too short ({len(audio)} bytes), ignoring')
            return
        self._set_stage('transcribing')
        start = time.time()
        text = self._transcribe(self._wav_bytes(audio))
        whisper_seconds = time.time() - start
        if len(text) < 3:
            self.get_logger().info(
                f'[HEARD] Whisper found no words ({text!r}) in {whisper_seconds:.1f} s; '
                'nothing sent to the LLM')
            self._log('audio', 'Kõnetuvastus ei leidnud sõnu, keelemudelile ei saadetud',
                      f'kõnetuvastus {whisper_seconds:.1f} s')
            return
        self.get_logger().info(f'[HEARD] {text} -> sending to LLM ({mode})')
        self._log('heard', text, f'kõnetuvastus {whisper_seconds:.1f} s, '
                                 f'saadetud: {MODE_NAMES[mode]}')

        backend = self.config.local_backend if mode == 'local' else self.config.cloud_backend
        if backend is not None:
            backend.last_usage = None
        self._set_stage('thinking')
        start = time.time()
        try:
            if backend is None:
                raise BackendError(self.config.cloud_error)
            reply = self.conversation.ask(backend, SYSTEM_PROMPT, text)
            self.get_logger().info(f'[LLM {mode}] replied in {time.time() - start:.1f}s')
        except BackendError as e:
            self.get_logger().error(f'[LLM {mode}] {e}')
            self._log('error', f'{MODE_NAMES[mode]}: {str(e)[:200]}')
            reply = APOLOGY
        finally:
            if mode == 'cloud' and backend is not None and backend.last_usage:
                self.usage.record(backend.last_usage, self.config.cloud_prices)
                self._publish_usage()
        llm_seconds = time.time() - start

        with self.lock:
            if self.mode != mode:
                self.get_logger().info('Mode changed while thinking, discarding reply')
                return
        self.get_logger().info(f'[SPEAKING] {reply}')
        start = time.time()
        wav_bytes = self._synthesize(reply)
        details = f'{MODE_NAMES[mode]} {llm_seconds:.1f} s · kõnesüntees {time.time() - start:.1f} s'
        self._log('said', reply, details)
        if wav_bytes:
            self._play(wav_bytes)

    @staticmethod
    def _wav_bytes(raw_audio):
        mem_file = io.BytesIO()
        with wave.open(mem_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_audio)
        return mem_file.getvalue()

    def _transcribe(self, wav_bytes):
        files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}
        try:
            start = time.time()
            response = requests.post(self.config.whisper_url, files=files, timeout=30)
            self.get_logger().info(f'[WHISPER] responded in {time.time() - start:.1f}s')
            response.raise_for_status()
            # Whisper puts line breaks between sentences; keep one line of text.
            return ' '.join(response.json().get('text', '').split())
        except (requests.exceptions.RequestException, ValueError) as e:
            self.get_logger().error(f'[WHISPER] {self.config.whisper_url}: {e}')
            self._log('error', f'Kõnetuvastus ebaõnnestus: {str(e)[:150]}')
            return ''

    def speak_text(self, text_to_speak):
        """Synthesizes speech via the TartuNLP TTS server and plays it."""
        wav_bytes = self._synthesize(text_to_speak)
        if wav_bytes:
            self._play(wav_bytes)

    def _synthesize(self, text):
        """WAV bytes from the TartuNLP TTS server, or None on failure."""
        self._set_stage('synthesizing')
        try:
            start_time = time.time()
            response = requests.post(
                self.config.tts_url,
                json={'text': text, **self.tts},
                timeout=300,
            )
            response.raise_for_status()
            wav_bytes = response.content
            if wav_bytes[:4] != b'RIFF' or wav_bytes[8:12] != b'WAVE':
                self.get_logger().error('[TTS] Server did not return a WAV file')
                return None
            self.get_logger().info(f'[TTS] Synthesized in {time.time() - start_time:.2f}s')
            return wav_bytes
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'[TTS] Request to {self.config.tts_url} failed: {e}')
            self._log('error', f'Kõnesüntees ebaõnnestus: {str(e)[:150]}')
            return None

    def _play(self, wav_bytes):
        self._set_stage('speaking')
        with tempfile.NamedTemporaryFile(suffix='.wav') as wav_file:
            wav_file.write(wav_bytes)
            wav_file.flush()
            try:
                subprocess.run(['play', '-q', wav_file.name], check=True,
                               capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f'[TTS] Playback failed ({e.returncode}): {e.stderr}')

def main(args=None):
    rclpy.init(args=args)
    node = SemuBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():  # SIGINT may already have shut the context down
            rclpy.shutdown()


if __name__ == '__main__':
    main()
