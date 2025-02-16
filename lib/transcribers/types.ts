export interface StreamAudioInput {
    type: 'stream';
    stream: ReadableStream;
}

export interface FileAudioInput {
    type: 'file';
    path: string | URL;
}

export type AudioInput = StreamAudioInput | FileAudioInput;


export interface Transcriber { 
    transcribe(input: AudioInput, options: TranscriberOptions): Promise<string>;
}

export interface TranscriberOptions {
    onProgress?: (textChunk: string) => void;
}

export type SpeechToTextModel = 'onnx-community/whisper-large-v3-turbo' |'openai/whisper-large-v3' | 'openai/whisper-medium' | 'openai/whisper-small' | 'openai/whisper-tiny';