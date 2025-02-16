import { assertEquals } from "@std/assert/equals";
import { AudioInput, SpeechToTextModel, Transcriber, TranscriberOptions } from "./types";
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

type Model = "onnx-community/whisper-large-v3-turbo" | "onnx-community/whisper-large-v3-turbo-q8_0"

export class WhisperCcliTranscriber implements Transcriber {
  constructor(private model: SpeechToTextModel) {}

  private async ensureWavFormat(filePath: string): Promise<string> {
    const probeCommand = spawn('ffprobe', [
      "-v",
      "quiet",
      "-print_format",
      "json",
      "-show_streams",
      filePath,
    ]);

    return new Promise((resolve, reject) => {
      let stdout = '';
      let stderr = '';

      probeCommand.stdout.on('data', (data) => {
        stdout += data;
      });

      probeCommand.stderr.on('data', (data) => {
        stderr += data;
      });

      probeCommand.on('close', async (code) => {
        if (code !== 0) {
          reject(new Error(`ffprobe failed: ${stderr}`));
          return;
        }

        const probeData = JSON.parse(stdout);
        const audioStream = probeData.streams.find((s: any) =>
          s.codec_type === "audio"
        );
        const needsConversion = !audioStream ||
          audioStream.sample_fmt !== "s16" ||
          audioStream.codec_name !== "pcm_s16le";

        if (!needsConversion) {
          resolve(filePath);
          return;
        }

        const outputPath = `${filePath}.wav`;
        const convertCommand = spawn('ffmpeg', [
          "-i",
          filePath,
          "-acodec",
          "pcm_s16le",
          "-ar",
          "16000",
          "-ac",
          "1",
          "-y",
          outputPath,
        ]);

        convertCommand.on('close', (code) => {
          if (code !== 0) {
            reject(new Error(`FFmpeg conversion failed`));
          } else {
            resolve(outputPath);
          }
        });
      });
    });
  }

  async transcribe(
    input: AudioInput,
    options: TranscriberOptions,
  ): Promise<string> {
    let filePath: string;
    if (input.type === "file") {
      filePath = input.path.toString();
    } else {
      const tempFile = path.join(os.tmpdir(), `whisper-${Date.now()}.tmp`);
      const writeStream = fs.createWriteStream(tempFile);
      await new Promise((resolve, reject) => {
        input.stream.pipeTo(
          new WritableStream({
            write(chunk) {
              writeStream.write(chunk);
            },
          })
        ).then(resolve).catch(reject);
      });
      filePath = tempFile;
    }

    const wavPath = await this.ensureWavFormat(filePath);

    const modelPath = "/Users/alangou/dev/@agou/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin";
    
    return new Promise((resolve, reject) => {
      let stdoutString = "";
      let stderrString = "";

      const command = spawn('whisper-cli', ["-m", modelPath, "-f", wavPath]);

      command.stdout.on('data', (chunk) => {
        const text = chunk.toString();
        stdoutString += text;
        const lines = parseLines(text);
        options.onProgress?.(lines.map((l) => l.text).join("\n"));
      });

      command.stderr.on('data', (chunk) => {
        stderrString += chunk.toString();
      });

      command.on('close', async (code) => {
        await fs.promises.writeFile("whisper-cli.stderr", stderrString);
        await fs.promises.writeFile("whisper-cli.stdout", stdoutString);

        if (code !== 0) {
          reject(new Error(`whisper-cli failed with code ${code}`));
          return;
        }

        // Cleanup temporary wav file if it was created
        if (wavPath !== filePath) {
          await fs.promises.unlink(wavPath);
        }

        resolve(stdoutString);
      });
    });
  }
}

const Regex = /^\[([0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}) --> ([0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3})\]\s*(.*?)$/;

interface WhisperCcLine {
  start: string;
  end: string;
  text: string;
  newSpeaker: boolean;
}

export function parseLines(text: string): WhisperCcLine[] {
  text = text.trim();
  return text.split("\n").map((line) => {
    const match = line.match(Regex);
    if (match) {
      const rawText = match[3].trim();
      const hasNewSpeaker = rawText.startsWith('-');
      return {
        start: match[1],
        end: match[2],
        text: rawText.replace(/^-\s*/, '').trim(),
        newSpeaker: hasNewSpeaker,
      };
    }
    return null;
  }).filter((s): s is WhisperCcLine => s != null && (s.text.trim() != "" || s.newSpeaker));
}

Deno.test("WhisperCcliTranscriber.parseLines", () => {
  const input = `
[00:00:00.000 --> 00:00:03.240]   -Good morning. This Tuesday is Election Day.
[00:00:03.240 --> 00:00:06.000]   After months of spirited debate and vigorous campaigning,
[00:00:06.000 --> 00:00:08.000]   -  
[00:00:08.000 --> 00:00:12.000]   The time has come to make our voices heard.
`;

  const expected = [
    {
      start: "00:00:00.000",
      end: "00:00:03.240",
      text: "Good morning. This Tuesday is Election Day.",
      newSpeaker: true,
    },
    {
      start: "00:00:03.240",
      end: "00:00:06.000",
      text: "After months of spirited debate and vigorous campaigning,",
      newSpeaker: false,
    },
    {
      start: "00:00:06.000",
      end: "00:00:08.000",
      text: "",
      newSpeaker: true,
    },
    {
      start: "00:00:08.000",
      end: "00:00:12.000",
      text: "The time has come to make our voices heard.",
      newSpeaker: false,
    },
  ];

  const result = parseLines(input);
  assertEquals(result, expected);
});
