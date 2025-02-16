import { pipeline } from "@huggingface/transformers";
import { AudioInput, SpeechToTextModel, Transcriber, TranscriberOptions } from "./types";
import * as fs from 'fs';
import { spawn } from 'child_process';
import * as os from 'os';
import * as path from 'path';

export class HuggingFaceTranscriber implements Transcriber {
    constructor(private model: SpeechToTextModel) {
    }

    async transcribe(input: AudioInput, options: TranscriberOptions): Promise<string> {
        let buffer: Float32Array;
        if (input.type === 'file') {
            const file = await fs.promises.readFile(input.path.toString());
            buffer = new Float32Array(file.buffer);
        } else {
            const stream = input.stream;
            const reader = stream.getReader();
            const result = await reader.read();
            buffer = new Float32Array(result.value);
        }

        const pipe = await makeSpeechToTextPipeline(this.model);
        const result = await pipe([buffer]);
        if (Array.isArray(result))  {
            const joined = result.map(r => r.text).join('\n');
            options.onProgress?.(joined);
            return joined;
        }
        options.onProgress?.(result.text);
        return result.text;
    }
}

function makeSpeechToTextPipeline(model: SpeechToTextModel = 'onnx-community/whisper-large-v3-turbo') {
    return pipeline('automatic-speech-recognition', model, {
        device: 'cpu',
        cache_dir: './cache',
        progress_callback: (info) => {
            switch (info.status) {
                case 'initiate':
                    console.log(`initiate: ${info.name} - ${info.file}`);
                    break;
                case 'download':
                    console.log(`download: ${info.name} - ${info.file}`);
                    break;
                case 'progress':
                    break;
                case 'done':
                    console.log(`done: ${info.name} - ${info.file} - ${info.status}`);
                    break;
                case 'ready':
                    console.log('ready')
                    break;
            }
        },
    });
}


