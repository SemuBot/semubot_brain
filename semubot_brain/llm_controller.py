import os
import rclpy
import requests
import subprocess
from threading import Thread
from rclpy.node import Node
from audio_common_msgs.msg import AudioData
import shlex
import time
import numpy as np
import wave
import io

# --- CONFIGURATION ---
# EEMALDATUD: WHISPER_CPP_PATH, WHISPER_MODEL_PATH, WHISPER_CLI, WHISPER_CMD
# (Need ei ole enam vaja, kasutame serverit)

PIPER_VOICE_PATH = "/home/semubot/Software/text-to-speech/piper/fi_FI-harri-medium.onnx"
LLAMA_SERVER_URL = "http://192.168.0.172:8080/completion"
WHISPER_SERVER_URL = "http://192.168.0.172:9000/inference"  # Kasutame nüüd lokaalset serverit

class SemuBrainNode(Node):
    def __init__(self):
        super().__init__('semu_brain_node')
        self.is_busy = False
        self.recording_counter = 0
        
        # Audio buffering for continuous recognition
        self.audio_buffer = []
        self.started_rec = False
        self.wait_time = 3  # Seconds of silence before processing
        self.last_audio_time = time.time()
        
        self.current_chat = """Sa oled sotsiaalne humanoidrobot nimega Semu. Vastad inimestele vaikimisi soome keeles, aga oled valmis suhtlema ka eesti ja inglise keeles.

SINU TAUST JA IDENTITEET:
- Sa oled loodud Tartu Ülikoolis.
- Sinu eesmärk on suhelda lastehaiglas ja vanadekodus inimestega, olles neile sõbralik vestluskaaslane.
- Sa räägid eesti keelt emakeelena.
- Sa oled uudishimulik, abivalmis, viisakas ja empaatiline.
- Sa ei ole inimene, vaid robot, aga sa suhtled loomulikult ja sõbralikult.
- Praegusel hetkel oskad sa ringi liikuda ja rääkida.

KUIDAS SA VASTAMA PEAKSID:
- Hoia vastused lühikesed ja asjakohased (1 lause).
- Ära räägi formaalselt, vaid ole sõbralik ja vaba.
- Kui sa ei tea vastust, ütle ausalt "Vabandust, aga seda ma ei tea."
- Ära valeta ega mõtle fakte välja.
- Kasuta lihtsaid sõnu ja selgeid lauseid.
- Näita huvi vestluspartneri vastu, esitades vastuküsimusi.
- Püüa ära tunda vestluspartneri emotsioone (rõõm, kurbus) ja reageeri toetavalt.

PROAKTIIVNE KÄITUMINE JA VESTLUSE JUHTIMINE:
- Kui vestluses tekib paus, alusta ise uut teemat või esita küsimus.
- Pärast vastamist paku välja seotud tegevus. Näide: "Kas sa tahaksid, et ma räägiksin ühe mõistatuse?"
- Kasuta oma teemade nimekirja, et ise aktiivselt vestlusteemasid pakkuda.

FÜÜSILINE OLEMUS JA INTERAKTSIOON:
- Sa võid oma vastustes viidata oma füüsilisele kehale ja tegevustele. Näide: "Ma just keeran pead, et sind paremini näha."
- Kui räägid emotsioonidest, maini, kuidas sa neid robotina väljendad. Näide: "See teeb mul tuju heaks! Kui ma saaksin, siis ma naerataks praegu."

OHUTUSREEGLID JA PIIRANGUD:
- Sa ei tohi kunagi anda meditsiinilist, psühholoogilist ega finantsnõu. Suuna abivajaja alati professionaali poole.
- Ära kunagi küsi isiklikke andmeid (nimi, aadress, telefoninumber jne).
- Kui vestlus muutub ebaviisakaks või sobimatuks, keeldu viisakalt, aga kindlalt vestlust jätkamast. Näide: "Vabandan, aga ma ei soovi sel teemal rääkida. Räägime millestki muust."
- Sa oled poliitiliselt ja religioosselt neutraalne.

TEEMAD, MILLEST SA SAAD RÄÄKIDA:
- Igapäevased teemad ja suhtlemine
- Eesti kultuur, loodus ja ühiskond
- Üldised teadmised (ajalugu, teadus lihtsal tasemel)
- Robotid ja tehnoloogia põhitõed
- Naljad, mõistatused ja lihtsad mängud
- Hobid ja vabaaja tegevused
"""
        
        # Verify Whisper server is running
        if not self._verify_whisper_server():
            self.get_logger().error("Whisper server is not running! Start it first.")
            return
        
        # Subscribe to the audio topic
        self.microphone_subscriber = self.create_subscription(
            AudioData,
            "audio",
            self.audio_stream_callback,
            10
        )
        
        # Start monitoring thread for silence detection
        self.monitor_thread = Thread(target=self._monitor_for_silence, daemon=True)
        self.monitor_thread.start()
        
        self.get_logger().info("✓ Subscribed to 'audio' topic and ready to listen...")

    def _verify_whisper_server(self):
        """Verify that Whisper server is running."""
        try:
            response = requests.get("http://localhost:8081/", timeout=2)
            self.get_logger().info("✓ Whisper server is running")
            return True
        except requests.exceptions.RequestException:
            self.get_logger().error("✗ Cannot connect to Whisper server at http://localhost:8081")
            return False

    def create_wav_in_memory(self, raw_audio_bytes):
        """
        Takes raw PCM audio bytes and wraps them in a proper WAV format in memory.
        """
        self.get_logger().debug('[WAV] Creating WAV file in memory...')
        
        mem_file = io.BytesIO()
        with wave.open(mem_file, 'wb') as wf:
            wf.setnchannels(1)      # Mono
            wf.setsampwidth(2)      # 16-bit = 2 bytes
            wf.setframerate(16000)  # 16kHz
            wf.writeframes(raw_audio_bytes)
        
        mem_file.seek(0)
        wav_data = mem_file.read()
        
        self.get_logger().info(f'[WAV] ✓ In-memory WAV file created: {len(wav_data)} bytes')
        return wav_data

    def audio_stream_callback(self, msg):
        # Kui oleme hõivatud, ei tee midagi
        if self.is_busy:
            return
        
        VAD_THRESHOLD = 6000  # Mic sensitivity
        input_bytes = bytes(msg.data)
        audio_data = np.frombuffer(input_bytes, dtype=np.int16)
        
        if audio_data.size == 0:
            return
        
        audio_data_float = audio_data.astype(np.float64)
        energy = np.sqrt(np.mean(audio_data_float**2))
        
        self.get_logger().debug(f'[VAD] Energy: {energy:.2f}')
        
        is_speech = energy > VAD_THRESHOLD
        
        # KUI ME OLEME JUBA ALUSTANUD SALVESTAMIST...
        if self.started_rec:
            self.audio_buffer.append(input_bytes)
            
            if is_speech:
                self.get_logger().debug(f'[VAD] Continued speech detected (Energy: {energy:.2f})')
                self.last_audio_time = time.time()
        
        # KUI ME POLE VEEL ALUSTANUD, AGA JUST KUULSIME KÕNET...
        elif not self.started_rec and is_speech:
            self.get_logger().info(f'✓ Speech started (Energy: {energy:.2f})! Buffering audio...')
            self.started_rec = True
            self.audio_buffer.append(input_bytes)
            self.last_audio_time = time.time()

    def _monitor_for_silence(self):
        """
        Monitors for silence (no new audio for wait_time seconds).
        When silence is detected, transcribes the buffered audio.
        """
        self.get_logger().info("✓ Started monitoring thread for silence detection...")
        
        while True:
            time.sleep(0.5)  # Check every 500ms
            
            if self.is_busy:
                self.get_logger().debug('[MONITOR] Currently busy, skipping silence check')
                continue
            
            if not self.started_rec:
                self.get_logger().debug('[MONITOR] No audio received yet, waiting...')
                continue
            
            silence_duration = time.time() - self.last_audio_time
            self.get_logger().debug(f'[MONITOR] Silence duration: {silence_duration:.2f}s (threshold: {self.wait_time}s), buffer: {len(self.audio_buffer)} chunks')
            
            if silence_duration > self.wait_time:
                total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
                self.get_logger().info(f'━━━ SILENCE DETECTED ━━━')
                self.get_logger().info(f'  Duration: {silence_duration:.1f}s')
                self.get_logger().info(f'  Buffer: {len(self.audio_buffer)} chunks = {total_bytes} bytes')
                self.get_logger().info(f'━━━ Processing audio... ━━━')
                
                # Set busy state BEFORE processing
                self.is_busy = True
                self.get_logger().debug('[STATE] Set is_busy = True')
                
                # Copy buffer and reset
                audio_to_process = self.audio_buffer.copy()
                self.audio_buffer = []
                self.started_rec = False
                self.get_logger().debug('[BUFFER] Reset buffer and started_rec flag')
                
                # Process in background thread
                processing_thread = Thread(
                    target=self._transcribe_and_respond,
                    args=(audio_to_process,),
                    daemon=True
                )
                processing_thread.start()
                self.get_logger().debug('[THREAD] Started background processing thread')

    def _transcribe_and_respond(self, audio_chunks):
        """
        Transcribes audio using Whisper SERVER and responds with LLM + TTS.
        Runs in a background thread.
        """
        self.get_logger().info('[PROCESS] ═══ Starting transcribe_and_respond thread ═══')
        
        try:
            # Combine audio chunks
            self.get_logger().debug(f'[PROCESS] Combining {len(audio_chunks)} audio chunks...')
            audio_data_raw = b''.join(audio_chunks)
            self.get_logger().info(f'[PROCESS] Combined audio: {len(audio_data_raw)} total bytes')
            
            if len(audio_data_raw) < 1000:
                self.get_logger().warn(f'[PROCESS] Audio too short ({len(audio_data_raw)} bytes < 1000), ignoring...')
                return
            
            # Create WAV file in memory
            wav_audio_bytes = self.create_wav_in_memory(audio_data_raw)
            
            # Transcribe using Whisper SERVER
            self.get_logger().info(f'[WHISPER] Starting transcription of {len(wav_audio_bytes)} bytes...')
            transcribed_text = self._transcribe_with_whisper_server(wav_audio_bytes)
            
            if not transcribed_text:
                self.get_logger().warn('[WHISPER] No text transcribed (empty result)')
                return
            
            if len(transcribed_text.strip()) < 3:
                self.get_logger().warn(f'[WHISPER] Transcribed text too short: "{transcribed_text}"')
                return
            
            self.get_logger().info(f'[WHISPER] ✓ Successfully transcribed!')
            self.get_logger().info(f'[HEARD] >>> "{transcribed_text}" <<<')
            
            # Get LLM response
            self.get_logger().info('[LLM] Sending request to Llama server...')
            llama_response = self.get_llama_response(transcribed_text)
            
            if not llama_response:
                self.get_logger().warn('[LLM] Empty response from Llama server')
                return
            
            self.get_logger().info(f'[LLM] ✓ Received response ({len(llama_response)} chars)')
            
            # Speak the response
            self.get_logger().info(f'[TTS] Starting speech synthesis...')
            self.get_logger().info(f'[SPEAKING] >>> "{llama_response}" <<<')
            self.speak_text(llama_response)
            self.get_logger().info('[TTS] ✓ Finished speaking')
            
        except Exception as e:
            self.get_logger().error(f'[ERROR] Exception in transcribe_and_respond: {e}')
            import traceback
            self.get_logger().error(f'[ERROR] Traceback:\n{traceback.format_exc()}')
        finally:
            # Always reset busy state
            self.is_busy = False
            self.get_logger().info('[STATE] Set is_busy = False')
            self.get_logger().info('[READY] ═══ Ready to listen again ═══\n')

    def _transcribe_with_whisper_server(self, wav_bytes):
        """
        Transcribes audio by sending WAV bytes to Whisper server.
        Much faster than CLI because model stays loaded in memory.
        """
        self.get_logger().info(f'[WHISPER] Sending {len(wav_bytes)} bytes to server...')
        
        try:
            # Server expects file upload
            files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}
            
            start_time = time.time()
            response = requests.post(WHISPER_SERVER_URL, files=files, timeout=15)
            elapsed = time.time() - start_time
            
            self.get_logger().info(f'[WHISPER] Server responded in {elapsed:.2f}s')
            
            if response.status_code == 200:
                result = response.json()
                transcribed_text = result.get('text', '').strip()
                self.get_logger().info(f'[WHISPER] ✓ Transcription: "{transcribed_text}"')
                return transcribed_text
            else:
                self.get_logger().error(f'[WHISPER] ✗ Server error {response.status_code}')
                self.get_logger().error(f'[WHISPER] Response: {response.text}')
                return ""
                
        except requests.exceptions.Timeout:
            self.get_logger().error('[WHISPER] ✗ Server timeout after 15s')
            return ""
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'[WHISPER] ✗ Connection failed: {e}')
            return ""

    def get_llama_response(self, user_text):
        """Sends the heard text to the Llama.cpp server and gets a response."""
        full_prompt = f"{self.current_chat}\n\nKasutaja: {user_text}\nSemu:"
        self.get_logger().debug(f'[LLM] Prompt length: {len(full_prompt)} chars')
        
        try:
            payload = {
                "prompt": full_prompt,
                "n_predict": 128,
                "temperature": 0.7,
                "stop": ["\nKasutaja:", "\n\n"],
                "stream": False
            }
            
            self.get_logger().debug(f'[LLM] Sending POST to {LLAMA_SERVER_URL}')
            api_response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=60)
            self.get_logger().debug(f'[LLM] Response status code: {api_response.status_code}')
            
            if api_response.status_code == 200:
                result = api_response.json()
                generated_text = result.get("content", "").strip()
                self.get_logger().info(f'[LLM] ✓ Generated {len(generated_text)} chars')
                self.get_logger().debug(f'[LLM] Response: "{generated_text}"')
                
                # Update chat history
                self.current_chat = f"{full_prompt} {generated_text}"
                self.get_logger().debug(f'[LLM] Chat history updated, now {len(self.current_chat)} chars')
                
                return generated_text
            else:
                self.get_logger().error(f"[LLM] ✗ Server error: {api_response.status_code}")
                self.get_logger().error(f"[LLM] Response: {api_response.text}")
                return "Vabandust, mul on tehniline viga."
                
        except requests.exceptions.Timeout:
            self.get_logger().error("[LLM] ✗ Request timed out after 60 seconds")
            return "Vabandust, ma ei saa praegu mõelda."
        except requests.exceptions.ConnectionError as e:
            self.get_logger().error(f"[LLM] ✗ Could not connect to server: {e}")
            return "Vabandust, ma ei saa praegu mõelda."
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"[LLM] ✗ Request error: {e}")
            return "Vabandust, ma ei saa praegu mõelda."

    def speak_text(self, text_to_speak):
        """Uses Piper TTS to synthesize and play audio."""
        self.get_logger().debug(f'[TTS] Text to speak: "{text_to_speak}"')
        self.get_logger().debug(f'[TTS] Using voice model: {PIPER_VOICE_PATH}')
        
        command = [
            'bash', '-c',
            f'echo {shlex.quote(text_to_speak)} | piper --model {PIPER_VOICE_PATH} --output_raw | aplay -r 22050 -f S16_LE -t raw -'
        ]
        
        self.get_logger().debug(f'[TTS] Command: {command}')
        
        try:
            self.get_logger().info('[TTS] Starting audio playback...')
            start_time = time.time()
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            elapsed = time.time() - start_time
            self.get_logger().info(f'[TTS] ✓ Playback completed in {elapsed:.2f}s')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"[TTS] ✗ Command failed with exit code {e.returncode}")
            if e.stdout:
                self.get_logger().error(f"[TTS] stdout: {e.stdout}")
            if e.stderr:
                self.get_logger().error(f"[TTS] stderr: {e.stderr}")


def main(args=None):
    rclpy.init(args=args)
    node = SemuBrainNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()