/**
 * Phase 18 & 19: Audio Engine
 * Handles microphone recording (16kHz PCM) and speaker playback (24kHz PCM).
 * Includes waveform visualizer data generation.
 */

export class AudioRecorder {
    constructor(onAudioChunk, onWaveformData) {
        this.onAudioChunk = onAudioChunk;
        this.onWaveformData = onWaveformData;
        this.stream = null;
        this.context = null;
        this.processor = null;
        this.analyser = null;
        this.isRecording = false;
        this._animFrame = null;
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });

            this.context = new AudioContext({ sampleRate: 16000 });
            const source = this.context.createMediaStreamSource(this.stream);

            // Analyser for waveform visualization
            this.analyser = this.context.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);

            // Processor for PCM data
            this.processor = this.context.createScriptProcessor(4096, 1, 1);
            source.connect(this.processor);
            this.processor.connect(this.context.destination);

            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                const float32 = e.inputBuffer.getChannelData(0);
                const pcm16 = this._floatTo16Bit(float32);
                this.onAudioChunk(pcm16);
            };

            this.isRecording = true;
            this._startWaveformLoop();
            return true;

        } catch (err) {
            console.error('Microphone error:', err);
            return false;
        }
    }

    stop() {
        this.isRecording = false;
        if (this._animFrame) cancelAnimationFrame(this._animFrame);
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        if (this.context) {
            this.context.close();
            this.context = null;
        }
        this.processor = null;
        this.analyser = null;
    }

    _floatTo16Bit(float32) {
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
            const s = Math.max(-1, Math.min(1, float32[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16;
    }

    _startWaveformLoop() {
        if (!this.analyser || !this.onWaveformData) return;
        const data = new Uint8Array(this.analyser.frequencyBinCount);

        const loop = () => {
            if (!this.isRecording) return;
            this.analyser.getByteFrequencyData(data);
            // Downsample to 20 bars
            const bars = [];
            const step = Math.floor(data.length / 20);
            for (let i = 0; i < 20; i++) {
                bars.push(data[i * step] / 255);
            }
            this.onWaveformData(bars);
            this._animFrame = requestAnimationFrame(loop);
        };
        loop();
    }
}


export class AudioPlayer {
    constructor() {
        this._queue = [];
        this._playing = false;
        this._context = null;
    }

    enqueue(pcm16ArrayBuffer) {
        this._queue.push(pcm16ArrayBuffer);
        this._playNext();
    }

    async _playNext() {
        if (this._playing || this._queue.length === 0) return;
        this._playing = true;

        if (!this._context) {
            this._context = new AudioContext({ sampleRate: 24000 });
        }
        if (this._context.state === 'suspended') {
            await this._context.resume();
        }

        while (this._queue.length > 0) {
            const pcmBuf = this._queue.shift();
            const float32 = this._pcm16ToFloat32(pcmBuf);

            if (float32.length === 0) continue;

            const buffer = this._context.createBuffer(1, float32.length, 24000);
            buffer.getChannelData(0).set(float32);

            const source = this._context.createBufferSource();
            source.buffer = buffer;
            source.connect(this._context.destination);
            source.start();

            await new Promise(resolve => { source.onended = resolve; });
        }

        this._playing = false;
    }

    stop() {
        this._queue = [];
        if (this._context) {
            this._context.close().catch(err => console.warn('AudioContext close error:', err));
            this._context = null;
        }
        this._playing = false;
    }

    _pcm16ToFloat32(arrayBuffer) {
        const int16 = new Int16Array(arrayBuffer);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768.0;
        }
        return float32;
    }
}
