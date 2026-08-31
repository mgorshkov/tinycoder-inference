// q35moe_cfg.cpp — dump full qwen35/qwen35moe config from GGUF (one-off diagnostic)
#include "GGUFLoader.hpp"
#include "ModelConfig.hpp"
#include <cstdio>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    tinycoder::GGUFLoader loader;
    if (!loader.loadMetadata(argv[1])) { std::fprintf(stderr, "load failed\n"); return 1; }
    const tinycoder::ModelConfig &c = loader.config();
    std::printf("arch=%s name=%s layers=%u hidden=%u\n", c.architecture.c_str(),
                c.modelName.c_str(), c.numLayers, c.hiddenSize);
    std::printf("nHeads=%u nKVHeads=%u headDim=%u attnKeyLen=%u attnValLen=%u\n",
                c.numAttentionHeads, c.numKVHeads, c.headDim,
                c.attentionKeyLength, c.attentionValueLength);
    std::printf("inter=%u ropeTheta=%g ropeDimCount=%u ropeSections=[%u,%u,%u,%u]\n",
                c.intermediateSize, c.ropeTheta, c.ropeDimensionCount,
                c.ropeDimensionSections[0], c.ropeDimensionSections[1],
                c.ropeDimensionSections[2], c.ropeDimensionSections[3]);
    std::printf("ssmInner=%u ssmState=%u ssmConvK=%u ssmGroups=%u ssmRank=%u\n",
                c.ssmInnerSize, c.ssmStateSize, c.ssmConvKernel,
                c.ssmGroupCount, c.ssmTimeStepRank);
    std::printf("fullAttnInterval=%u nextn=%u maxSeqLen=%u vocab=%u\n",
                c.fullAttentionInterval, c.nextnPredictLayers, c.maxSeqLen,
                c.vocabSize);
    std::printf("expertCount=%u expertUsed=%u expertFF=%u sharedFF=%u\n",
                c.expertCount, c.expertUsedCount, c.expertFeedForwardLength,
                c.expertSharedFeedForwardLength);
    return 0;
}
