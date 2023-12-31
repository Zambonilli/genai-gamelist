import * as fsPromises from "fs/promises";
import fs from "fs";
import { pipeline } from "node:stream/promises";
import path from "path";
import { program } from "commander";
import xml2js from "xml2js";
import {
  LlamaModel,
  LlamaContext,
  LlamaChatSession,
  LlamaJsonSchemaGrammar,
} from "node-llama-cpp";
// @ts-ignore
import { DiffusionPipeline } from "@aislamov/diffusers.js";
import { PNG } from "pngjs";

// TODO figure out why LlamaJsonSchemaGrammar ctor is defined w/an arg
// @ts-ignore
const gameResponseSchema = new LlamaJsonSchemaGrammar({
  type: "object",
  properties: {
    name: {
      type: "string",
    },
    desc: {
      type: "string",
    },
    rating: {
      type: "number",
    },
    releasedate: {
      type: "string",
    },
    developer: {
      type: "string",
    },
    publisher: {
      type: "string",
    },
    genre: {
      type: "string",
    },
    players: {
      type: "number",
    },
  },
});

const setupOutDir = async (outDir: string) => {
  try {
    const statResults = await fsPromises.stat(outDir);

    if (statResults.isDirectory()) {
      console.log(`deleting existing output at ${outDir}.`);
      await fsPromises.rm(outDir, { recursive: true });
      console.log(`deleted existing output at ${outDir}.`);
    }
  } catch {
    // does not exist
  }

  await fsPromises.mkdir(outDir);
  await fsPromises.mkdir(path.join(outDir, "media"));
  await fsPromises.mkdir(path.join(outDir, "media", "images"));
  console.log(`created outDir ${outDir}.`);
};

(async function main() {
  try {
    program
      .option("-m, --modelPath <modelPath>", "path to your llama2 guff file")
      .option("-i, --inputDir <inputDir>", "path to a directory with rom files")
      .option("-o, --outDir <outDir>", "path to a directory for the results")
      .option(
        "-g, --gpuLayers <gpuLayers>",
        "number of gpu layers to use with llama2",
        "1",
      )
      .action(
        async (options: {
          modelPath: string;
          inputDir: string;
          outDir: string;
          gpuLayers: string;
        }) => {
          const { modelPath, inputDir, outDir, gpuLayers } = options;

          await setupOutDir(outDir);

          console.log(`loading model from ${modelPath}.`);
          const model = new LlamaModel({
            modelPath,
            gpuLayers: parseInt(gpuLayers),
          });
          const context = new LlamaContext({
            model,
            contextSize: 4096,
            //batchSize: 528,
          });
          const session = new LlamaChatSession({
            contextSequence: context.getSequence(),
            systemPrompt: `
            Create a JSON document with the following fields and values for a \
            North American Sega Genesis game rom file.
          
            name: string, the displayed name for the game
            desc: string, a description of the game including any media description released, characters, plot points, goals, etc
            rating: float, the rating for the game, expressed as a floating point number between 0 and 1. Arbitrary values are fine (ES can display half-stars, quarter-stars, etc).
            releasedate: datetime, the date the game was released. Displayed as date only, time is ignored.
            developer: string, the development studio that created the game.
            publisher: string, the company that published the game.
            genre: string, the (primary) genre for the game.
            players: integer, the number of players the game supports.
          `,
          });
          console.log(`loaded model from ${modelPath}.`);

          const results: {
            gameList: {
              game: {
                path?: string;
                name: string;
                desc: string;
                image?: string;
                thumbnail?: string;
                rating: number;
                releasedate: string;
                developer: string;
                publisher: string;
                genre: string;
                players: number;
              };
            }[];
          } = { gameList: [] };

          console.log(`reading rom files from ${inputDir}`);
          const files = (await fsPromises.readdir(inputDir)).filter((f) =>
            f.endsWith(".zip"),
          );
          console.log(`read ${files.length} rom files from ${inputDir}`);

          for (const file of files) {
            try {
              console.log(`starting ${file}`);

              const response = await session.prompt(`rom file "${file}"`, {
                // @ts-ignore
                grammar: gameResponseSchema,
              });
              const game = JSON.parse(response);
              game.path = `/home/pi/ROMs/genesis/${file}`;
              results.gameList.push({ game });
            } catch (err: any) {
              console.error(`error: ${err.message} ${err.stack}`);
            } finally {
              console.log(`finished ${file}`);
            }
          }

          // explicitly free up the model so we get cpu/gpu memory reclaimed
          // TODO watch for destructor support
          model.dispose();

          console.log("starting image generation");

          const pipe = await DiffusionPipeline.fromPretrained(
            "aislamov/stable-diffusion-2-1-base-onnx",
            {
              revision: "cpu",
            },
          );
          for (const game of results.gameList) {
            const images = await pipe.run({
              prompt: `An box cover art image for the Sega Genesis game ${game.game.name} about ${game.game.desc}`,
              negativePrompt: "",
              numInferenceSteps: 30,
              sdV1: false,
              height: 768,
              width: 768,
              guidanceScale: 7.5,
              img2imgFlag: false,
              progressCallback: (progress: any) => {
                console.log(`${progress.statusText}`);
              },
            });

            const data = await images[0]
              .mul(255)
              .round()
              .clipByValue(0, 255)
              .transpose(0, 2, 3, 1);
            const p = new PNG({ width: 512, height: 512, inputColorType: 2 });
            p.data = Buffer.from(data.data);

            const imgPath = path.join(
              outDir,
              "media",
              "images",
              `${game.game.name.toLowerCase()}.png`,
            );
            await pipeline(p.pack(), fs.createWriteStream(imgPath));
            console.log(`image saved for ${game.game.name} to ${imgPath}`);

            game.game.image = `./media/images/${game.game.name}.png`;
          }

          // write file
          const outputFilePath = path.join(outDir, "gamelist.xml");
          const builder = new xml2js.Builder();
          await fsPromises.writeFile(
            path.join(outDir, "gamelist.xml"),
            builder.buildObject(results),
            "utf-8",
          );

          console.log(`wrote ${outputFilePath} file`);
        },
      );

    await program.parseAsync(process.argv);

    console.log("finished");
    process.exit(0);
  } catch (err: any) {
    console.error(`fatal error ${err.message} ${err.stack}`);
    process.exit(1);
  }
})();
