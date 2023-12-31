# genai-gamelist
genai-gamelist is a Command Line tool for creating retropie gamelist.xml and game resources using llama2 and stable diffusion.

## Local Development

### Setup

1. clone this repo
2. install nodejs 18.x
3. npm install
4. **optional** compile node-llama-cpp for [CUDA](https://withcatai.github.io/node-llama-cpp/guide/CUDA)
5. download meta's [llama2](https://ai.meta.com/llama/) LLM
6. convert llama2's to gguf file using [llama-cpp](https://github.com/ggerganov/llama.cpp#prepare-data--run)

### Building

```bash
npm run format
npm run build
```

### Running

```bash
node dist/index.js \
    --modelPath <modelPath> \
    --inputDir <inputDir> \
    --outDir <outDir>
```

### Debugging

A vscode debugging file is pushed with this repo. The `Launch Program` profile will launch the cli with default switches and allow you to debug the cli.