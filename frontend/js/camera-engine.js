/**
 * Phase 20: Camera capture engine.
 * Captures frames from the user's camera at ~1 FPS and sends as JPEG.
 */

export class CameraEngine {
    constructor(videoElement, onFrame) {
        this.video = videoElement;
        this.onFrame = onFrame;
        this.stream = null;
        this.interval = null;
        this.canvas = document.createElement('canvas');
        this.canvas.width = 640;
        this.canvas.height = 480;
        this.ctx = this.canvas.getContext('2d');
        this.active = false;
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'environment' }
            });
            this.video.srcObject = this.stream;
            this.active = true;

            // Capture at 1 FPS
            this.interval = setInterval(() => this._captureFrame(), 1000);
            return true;

        } catch (err) {
            console.error('Camera error:', err);
            return false;
        }
    }

    stop() {
        this.active = false;
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        if (this.video) {
            this.video.srcObject = null;
        }
    }

    _captureFrame() {
        if (!this.active || !this.video.videoWidth) return;

        this.ctx.drawImage(this.video, 0, 0, 640, 480);
        this.canvas.toBlob(
            (blob) => {
                if (!blob || !this.active) return;
                blob.arrayBuffer().then(buf => this.onFrame(buf));
            },
            'image/jpeg',
            0.7
        );
    }
}
