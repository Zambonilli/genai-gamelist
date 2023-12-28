import * as fs from "fs/promises";
import path from "path";
import { program } from "commander";
import xml2js from "xml2js";
import {
  LlamaModel,
  LlamaContext,
  LlamaChatSession,
  LlamaJsonSchemaGrammar,
} from "node-llama-cpp";

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
    const statResults = await fs.stat(outDir);

    if (statResults.isDirectory()) {
      console.log(`deleting existing output at ${outDir}.`);
      await fs.rm(outDir, { recursive: true });
      console.log(`deleted existing output at ${outDir}.`);
    }
  } catch {
    // does not exist
  }

  await fs.mkdir(outDir);
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
                name: string;
                desc: string;
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
          const files = (await fs.readdir(inputDir)).filter((f) =>
            f.endsWith(".zip"),
          );
          console.log(`read ${files.length} rom files from ${inputDir}`);

          for (const file of files) {
            try {
              console.log(`starting ${file}`);

              const response = await session.prompt(
                `rom file "${file}"`,
                {
                  // @ts-ignore
                  grammar: gameResponseSchema,
                },
              );
              results.gameList.push({ game: JSON.parse(response) });
            } catch (err: any) {
              console.error(`error: ${err.message} ${err.stack}`);
            } finally {
              console.log(`finished ${file}`);
            }
          }

          // write file
          const builder = new xml2js.Builder();
          await fs.writeFile(
            path.join(outDir, "gamelist.xml"),
            builder.buildObject(results),
            "utf-8",
          );

          console.log(`wrote file`);
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
