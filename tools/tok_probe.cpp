// tok_probe.cpp
//
// Dump TinyCoder's tokenizer output for a given prompt string WITHOUT loading
// the model weights (fast — GGUF metadata + embedded tokenizer only). Prints
// the exact same prompt that Model::generate() would feed the engine:
//   formatChat(system+user, true) -> tokenize(prompt)
//
// Build:
//   g++ -O2 -std=c++17 tools/tok_probe.cpp -I include -Isrc/cpp/core \
//       build/libtinycoder_core.a -lpthread -o /tmp/tok_probe
// Run:
//   /tmp/tok_probe <model.gguf> "<question text>"
//   /tmp/tok_probe <model.gguf> --write-wrapped <outfile>
//   /tmp/tok_probe <model.gguf> --encode-file <file> [--addspecial]
//   (the "assistant" header and system turn are hardcoded to match the test)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "GGUFLoader.hpp"
#include "ModelConfig.hpp"
#include "Tokenizer.hpp"

using namespace tinycoder;

static std::string buildWrapped(const std::string &question) {
    std::string prompt =
            "<|im_start|>system\nYou are TinyCoder, an AI coding assistant. Be concise.<|im_end|>\n"
            "<|im_start|>user\n" + question + "<|im_end|>\n"
            "<|im_start|>assistant\n";
    return "<|im_start|>user\n$" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

static std::string buildUnwrapped(const std::string &question) {
    return "<|im_start|>system\nYou are TinyCoder, an AI coding assistant. Be concise.<|im_end|>\n"
           "<|im_start|>user\n" + question + "<|im_end|>\n"
           "<|im_start|>assistant\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [\"<question>\"] | --write-wrapped <out> | "
                "--encode-file <file> [--addspecial]\n",
                argv[0]);
        return 1;
    }
    const char *modelPath = argv[1];

    GGUFLoader loader;
    if (!loader.loadMetadata(modelPath)) {
        fprintf(stderr, "metadata load failed\n");
        return 2;
    }
    const ModelConfig &cfg = loader.config();
    printf("  architecture=%s hidden=%u layers=%u\n",
           cfg.architecture.c_str(), cfg.hiddenSize, cfg.numLayers);

    Tokenizer tok;
    // Production order in Model::load(): configureForArchitecture FIRST, then
    // loadFromGGUF (which overrides BOS/EOS/PAD from metadata but does NOT
    // re-resolve im_start/im_end from the vocab).
    tok.configureForArchitecture(cfg.architecture);
    if (!tok.loadFromGGUF(modelPath)) {
        fprintf(stderr, "tokenizer load failed\n");
        return 2;
    }

    if (argc >= 3 && strcmp(argv[2], "--write-wrapped") == 0 && argc >= 4) {
        const std::string question =
                argc >= 5 ? argv[4]
                          : "What is the capital of France?";
        const std::string w = buildWrapped(question);
        std::ofstream out(argv[3], std::ios::binary);
        out.write(w.data(), (std::streamsize) w.size());
        out.close();
        printf("  [write-wrapped] %zu bytes -> %s\n", w.size(), argv[3]);
        return 0;
    }

    if (argc >= 3 && strcmp(argv[2], "--write-unwrapped") == 0 && argc >= 4) {
        const std::string question =
                argc >= 5 ? argv[4]
                          : "What is the capital of France?";
        const std::string u = buildUnwrapped(question);
        std::ofstream out(argv[3], std::ios::binary);
        out.write(u.data(), (std::streamsize) u.size());
        out.close();
        printf("  [write-unwrapped] %zu bytes -> %s\n", u.size(), argv[3]);
        return 0;
    }

    if (argc >= 3 && strcmp(argv[2], "--encode-file") == 0 && argc >= 4) {
        std::ifstream in(argv[3], std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
        std::vector<int32_t> t = tok.encode(data);
        printf("  [encode-file] %s (%zu bytes) -> n=%zu:\n", argv[3], data.size(), t.size());
        for (int32_t id : t) {
            std::string piece = tok.decodeToken(id);
            printf("    %d(%s%s)\n", id, piece.c_str(),
                   tok.isSpecialToken(id) ? "!" : "");
        }
        return 0;
    }

    // Reproduce Model::formatChat() for qwen35moe: system + user + assistant
    // header.  Note: with the llama-completion "$" wrap applied by formatChat
    // for qwen35moe, the actual string fed to tokenize() is:
    //   <|im_start|>user\n$<system+user turns><|im_end|>\n<|im_start|>assistant\n
    const std::string question =
            argc >= 3 ? argv[2] : "What is the capital of France?";
    std::string prompt =
            "<|im_start|>system\nYou are TinyCoder, an AI coding assistant. Be concise.<|im_end|>\n"
            "<|im_start|>user\n" + question + "<|im_end|>\n"
            "<|im_start|>assistant\n";
    std::string wrapped = buildWrapped(question);

    const char *label = (cfg.architecture == ARCH_QWEN35MOE ||
                         cfg.architecture == ARCH_QWEN35 ||
                         cfg.architecture == ARCH_QWEN2)
                                ? (cfg.chatTemplate.empty() ? "unwrapped" : "wrapped")
                                : "unwrapped";
    // For qwen35moe/qwen35 the chat template is non-empty, so formatChat SHIPS
    // the wrapped form to generate(). Print BOTH forms for comparison.
    printf("  vocabSize=%zu bos=%d eos=%d im_start=%d im_end=%d chatTemplate%s\n",
           tok.vocabSize(), tok.bosTokenId(), tok.eosTokenId(),
           tok.imStartId(), tok.imEndId(),
           cfg.chatTemplate.empty() ? "=EMPTY" : "=non-empty");

    const char *samples[] = {
            "assistant", "user", "system",
            "<|im_start|>assistant\n", "<|im_end|>\n<|im_start|>assistant\n"};
    printf("  [single] sample encodings:\n");
    for (const char *s : samples) {
        std::vector<int32_t> t = tok.encode(s);
        printf("    \"%s\" (len=%zu) ->", s, strlen(s));
        for (int32_t id : t) {
            std::string piece = tok.decodeToken(id);
            printf(" %d(%s%s)", id, piece.c_str(),
                   tok.isSpecialToken(id) ? "!" : "");
        }
        printf("\n");
    }

    printf("  [prompt] question=\"%s\"\n", question.c_str());
    printf("  [prompt] unwrapped (%zu bytes):\n", prompt.size());
    {
        std::vector<int32_t> t = tok.encode(prompt);
        printf("    encode n=%zu:", t.size());
        for (int32_t id : t) {
            std::string piece = tok.decodeToken(id);
            printf(" %d(%s%s)", id, piece.c_str(), tok.isSpecialToken(id) ? "!" : "");
        }
        printf("\n");
    }
    printf("  [prompt] wrapped (formatChat output for qwen35moe, %zu bytes):\n",
           wrapped.size());
    {
        std::vector<int32_t> t = tok.encode(wrapped);
        printf("    encode n=%zu:", t.size());
        for (int32_t id : t) {
            std::string piece = tok.decodeToken(id);
            printf(" %d(%s%s)", id, piece.c_str(), tok.isSpecialToken(id) ? "!" : "");
        }
        printf("\n");
    }
    return 0;
}