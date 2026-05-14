#include "tce_nnue_loader.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    TceNnueFileHeader header;
    TceNnueTensorInfo tensors[TCE_NNUE_TENSOR_COUNT];
    unsigned char checksum[TCE_NNUE_CHECKSUM_SIZE];
    char *metadata_json;
    unsigned char *file_bytes;
    size_t file_size;
    size_t payload_offset;
    size_t payload_size;
    int loaded;
} TceNnueLoadedFile;

static TceNnueLoadedFile g_nnue;

static int debug_enabled(void)
{
#ifdef TCE_NNUE_DEBUG
    return 1;
#else
    const char *value = getenv("TCE_NNUE_DEBUG");
    return value && value[0] && strcmp(value, "0") != 0;
#endif
}

static void debug_log(const char *message)
{
    if (debug_enabled())
        fprintf(stderr, "tce_nnue: %s\n", message);
}

static uint32_t read_le_u32(const unsigned char *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t read_le_u64(const unsigned char *p)
{
    uint64_t value = 0;
    for (int i = 7; i >= 0; i--)
        value = (value << 8) | p[i];
    return value;
}

static double read_le_double(const unsigned char *p)
{
    uint64_t bits = read_le_u64(p);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int read_file(const char *path, unsigned char **bytes, size_t *size)
{
    FILE *file = fopen(path, "rb");
    long file_size;
    unsigned char *buffer;

    if (!file) {
        fprintf(stderr, "tce_nnue: failed to open %s: %s\n", path, strerror(errno));
        return 1;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "tce_nnue: failed to seek %s\n", path);
        fclose(file);
        return 1;
    }

    file_size = ftell(file);
    if (file_size < 0) {
        fprintf(stderr, "tce_nnue: failed to measure %s\n", path);
        fclose(file);
        return 1;
    }
    rewind(file);

    buffer = (unsigned char *)malloc((size_t)file_size);
    if (!buffer) {
        fprintf(stderr, "tce_nnue: out of memory reading %s\n", path);
        fclose(file);
        return 1;
    }

    if (fread(buffer, 1, (size_t)file_size, file) != (size_t)file_size) {
        fprintf(stderr, "tce_nnue: failed to read %s\n", path);
        free(buffer);
        fclose(file);
        return 1;
    }

    fclose(file);
    *bytes = buffer;
    *size = (size_t)file_size;
    return 0;
}

static int parse_header(const unsigned char *bytes, size_t size, TceNnueFileHeader *header)
{
    if (size < TCE_NNUE_HEADER_SIZE + TCE_NNUE_CHECKSUM_SIZE) {
        fprintf(stderr, "tce_nnue: file too small for header and checksum\n");
        return 1;
    }

    memcpy(header->magic, bytes, TCE_NNUE_MAGIC_SIZE);
    header->format_version = read_le_u32(bytes + 8);
    header->feature_count = read_le_u32(bytes + 12);
    header->half_dim = read_le_u32(bytes + 16);
    header->hidden1_dim = read_le_u32(bytes + 20);
    header->hidden2_dim = read_le_u32(bytes + 24);
    header->output_dim = read_le_u32(bytes + 28);
    header->target_scale = read_le_double(bytes + 32);
    header->tensor_count = read_le_u32(bytes + 40);
    header->metadata_size = read_le_u64(bytes + 44);

    if (memcmp(header->magic, TCE_NNUE_MAGIC, strlen(TCE_NNUE_MAGIC)) != 0 ||
        header->magic[7] != '\0') {
        fprintf(stderr, "tce_nnue: invalid magic bytes\n");
        return 1;
    }

    if (header->format_version != TCE_NNUE_FORMAT_VERSION) {
        fprintf(stderr, "tce_nnue: unsupported format version %u\n",
                header->format_version);
        return 1;
    }

    if (header->tensor_count != TCE_NNUE_TENSOR_COUNT) {
        fprintf(stderr, "tce_nnue: expected %u tensors, found %u\n",
                TCE_NNUE_TENSOR_COUNT, header->tensor_count);
        return 1;
    }

    if (header->metadata_size > SIZE_MAX - TCE_NNUE_HEADER_SIZE) {
        fprintf(stderr, "tce_nnue: metadata size overflows platform size\n");
        return 1;
    }

    if (TCE_NNUE_HEADER_SIZE + (size_t)header->metadata_size >
        size - TCE_NNUE_CHECKSUM_SIZE) {
        fprintf(stderr, "tce_nnue: metadata extends beyond file payload\n");
        return 1;
    }

    return 0;
}

static const char *skip_ws(const char *p)
{
    while (*p && isspace((unsigned char)*p))
        p++;
    return p;
}

static int json_string_value(const char *object, const char *key, char *out, size_t out_size)
{
    char pattern[64];
    const char *p;
    const char *end;
    size_t len;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    p = strstr(object, pattern);
    if (!p)
        return 1;
    p += strlen(pattern);
    p = strchr(p, ':');
    if (!p)
        return 1;
    p = skip_ws(p + 1);
    if (*p != '"')
        return 1;
    p++;
    end = strchr(p, '"');
    if (!end)
        return 1;
    len = (size_t)(end - p);
    if (len >= out_size)
        len = out_size - 1;
    memcpy(out, p, len);
    out[len] = '\0';
    return 0;
}

static int json_u64_value(const char *object, const char *key, uint64_t *out)
{
    char pattern[64];
    const char *p;
    char *end;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    p = strstr(object, pattern);
    if (!p)
        return 1;
    p += strlen(pattern);
    p = strchr(p, ':');
    if (!p)
        return 1;
    p = skip_ws(p + 1);
    errno = 0;
    *out = strtoull(p, &end, 10);
    if (errno || end == p)
        return 1;
    return 0;
}

static int parse_shape(const char *object, TceNnueTensorInfo *tensor)
{
    const char *p = strstr(object, "\"shape\"");
    uint32_t rank = 0;

    if (!p)
        return 1;
    p = strchr(p, '[');
    if (!p)
        return 1;
    p++;

    while (*p && *p != ']') {
        char *end;
        unsigned long value;
        if (rank >= 4)
            return 1;
        p = skip_ws(p);
        errno = 0;
        value = strtoul(p, &end, 10);
        if (errno || end == p || value > UINT_MAX)
            return 1;
        tensor->shape[rank++] = (uint32_t)value;
        p = skip_ws(end);
        if (*p == ',')
            p++;
    }

    if (*p != ']')
        return 1;
    tensor->rank = rank;
    return 0;
}

static int parse_tensor_object(const char *object, TceNnueTensorInfo *tensor)
{
    uint64_t offset;
    uint64_t size;

    memset(tensor, 0, sizeof(*tensor));
    if (json_string_value(object, "name", tensor->name, sizeof(tensor->name)) ||
        json_string_value(object, "dtype", tensor->dtype, sizeof(tensor->dtype)) ||
        json_u64_value(object, "offset", &offset) ||
        json_u64_value(object, "size", &size) ||
        parse_shape(object, tensor)) {
        return 1;
    }

    tensor->offset = offset;
    tensor->size = size;
    return 0;
}

static int parse_tensors(char *metadata, TceNnueTensorInfo tensors[TCE_NNUE_TENSOR_COUNT])
{
    char *p = strstr(metadata, "\"tensors\"");
    if (!p) {
        fprintf(stderr, "tce_nnue: metadata missing tensors array\n");
        return 1;
    }

    p = strchr(p, '[');
    if (!p)
        return 1;
    p++;

    for (uint32_t i = 0; i < TCE_NNUE_TENSOR_COUNT; i++) {
        char *start = strchr(p, '{');
        char *end;
        char saved;
        if (!start)
            return 1;
        end = strchr(start, '}');
        if (!end)
            return 1;

        saved = end[1];
        end[1] = '\0';
        if (parse_tensor_object(start, &tensors[i])) {
            end[1] = saved;
            fprintf(stderr, "tce_nnue: failed to parse tensor metadata %u\n", i);
            return 1;
        }
        end[1] = saved;
        p = end + 1;
    }

    return 0;
}

static int expected_dtype(const char *name, const char *dtype)
{
    if (strstr(name, "bias"))
        return strcmp(dtype, "int32") == 0;
    return strcmp(dtype, "int16") == 0;
}

static int verify_tensors(const TceNnueTensorInfo tensors[TCE_NNUE_TENSOR_COUNT],
                          size_t payload_size)
{
    for (uint32_t i = 0; i < TCE_NNUE_TENSOR_COUNT; i++) {
        if (strcmp(tensors[i].name, TCE_NNUE_TENSOR_NAMES[i]) != 0) {
            fprintf(stderr, "tce_nnue: tensor %u expected %s, found %s\n",
                    i, TCE_NNUE_TENSOR_NAMES[i], tensors[i].name);
            return 1;
        }

        if (!expected_dtype(tensors[i].name, tensors[i].dtype)) {
            fprintf(stderr, "tce_nnue: tensor %s has unexpected dtype %s\n",
                    tensors[i].name, tensors[i].dtype);
            return 1;
        }

        if (tensors[i].offset > payload_size ||
            tensors[i].size > payload_size - (size_t)tensors[i].offset) {
            fprintf(stderr, "tce_nnue: tensor %s is outside payload bounds\n",
                    tensors[i].name);
            return 1;
        }
    }

    return 0;
}

/*
 * Minimal SHA256 implementation for loader validation.
 * Based on the FIPS 180-4 round function and kept local to avoid dependencies.
 */
typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    unsigned char data[64];
    size_t datalen;
} Sha256Ctx;

static const uint32_t k256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static uint32_t rotr(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32u - n));
}

static void sha256_transform(Sha256Ctx *ctx, const unsigned char data[64])
{
    uint32_t a, b, c, d, e, f, g, h, t1, t2, m[64];

    for (uint32_t i = 0, j = 0; i < 16; i++, j += 4)
        m[i] = ((uint32_t)data[j] << 24) |
               ((uint32_t)data[j + 1] << 16) |
               ((uint32_t)data[j + 2] << 8) |
               ((uint32_t)data[j + 3]);

    for (uint32_t i = 16; i < 64; i++) {
        uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
        uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
        m[i] = m[i - 16] + s0 + m[i - 7] + s1;
    }

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];

    for (uint32_t i = 0; i < 64; i++) {
        uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        t1 = h + s1 + ch + k256[i] + m[i];
        t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(Sha256Ctx *ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667u; ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u; ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu; ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu; ctx->state[7] = 0x5be0cd19u;
}

static void sha256_update(Sha256Ctx *ctx, const unsigned char *data, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

static void sha256_final(Sha256Ctx *ctx, unsigned char hash[32])
{
    size_t i = ctx->datalen;

    ctx->data[i++] = 0x80;
    if (i > 56) {
        while (i < 64)
            ctx->data[i++] = 0;
        sha256_transform(ctx, ctx->data);
        i = 0;
    }
    while (i < 56)
        ctx->data[i++] = 0;

    ctx->bitlen += ctx->datalen * 8;
    for (int j = 7; j >= 0; j--)
        ctx->data[63 - j] = (unsigned char)(ctx->bitlen >> (j * 8));
    sha256_transform(ctx, ctx->data);

    for (i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++)
            hash[i + j * 4] = (unsigned char)(ctx->state[j] >> (24 - i * 8));
    }
}

static int verify_checksum(const unsigned char *bytes, size_t size)
{
    unsigned char hash[32];
    Sha256Ctx ctx;

    if (size < TCE_NNUE_CHECKSUM_SIZE)
        return 1;

    sha256_init(&ctx);
    sha256_update(&ctx, bytes, size - TCE_NNUE_CHECKSUM_SIZE);
    sha256_final(&ctx, hash);

    if (memcmp(hash, bytes + size - TCE_NNUE_CHECKSUM_SIZE, 32) != 0) {
        fprintf(stderr, "tce_nnue: trailing SHA256 checksum mismatch\n");
        return 1;
    }

    return 0;
}

int tce_nnue_loader_load(const char *path)
{
    unsigned char *bytes = NULL;
    size_t size = 0;
    TceNnueLoadedFile next;
    size_t metadata_offset = TCE_NNUE_HEADER_SIZE;

    memset(&next, 0, sizeof(next));

    if (!path) {
        fprintf(stderr, "tce_nnue: null path\n");
        return 1;
    }

    if (read_file(path, &bytes, &size))
        return 1;

    if (parse_header(bytes, size, &next.header)) {
        free(bytes);
        return 1;
    }

    next.file_bytes = bytes;
    next.file_size = size;
    next.payload_offset = metadata_offset + (size_t)next.header.metadata_size;
    next.payload_size = size - next.payload_offset - TCE_NNUE_CHECKSUM_SIZE;
    memcpy(next.checksum, bytes + size - TCE_NNUE_CHECKSUM_SIZE, TCE_NNUE_CHECKSUM_SIZE);

    next.metadata_json = (char *)malloc((size_t)next.header.metadata_size + 1);
    if (!next.metadata_json) {
        fprintf(stderr, "tce_nnue: out of memory for metadata\n");
        free(bytes);
        return 1;
    }
    memcpy(next.metadata_json, bytes + metadata_offset, (size_t)next.header.metadata_size);
    next.metadata_json[next.header.metadata_size] = '\0';

    if (parse_tensors(next.metadata_json, next.tensors) ||
        verify_tensors(next.tensors, next.payload_size) ||
        verify_checksum(bytes, size)) {
        free(next.metadata_json);
        free(bytes);
        return 1;
    }

    if (debug_enabled()) {
        fprintf(stderr,
                "tce_nnue: loaded %s feature_count=%u half_dim=%u hidden=%u/%u payload=%zu\n",
                path,
                next.header.feature_count,
                next.header.half_dim,
                next.header.hidden1_dim,
                next.header.hidden2_dim,
                next.payload_size);
        for (uint32_t i = 0; i < TCE_NNUE_TENSOR_COUNT; i++) {
            fprintf(stderr, "tce_nnue: tensor %s dtype=%s offset=%llu size=%llu\n",
                    next.tensors[i].name,
                    next.tensors[i].dtype,
                    (unsigned long long)next.tensors[i].offset,
                    (unsigned long long)next.tensors[i].size);
        }
    } else {
        debug_log("loaded tcennue file");
    }

    tce_nnue_loader_free();
    g_nnue = next;
    g_nnue.loaded = 1;
    return 0;
}

void tce_nnue_loader_free(void)
{
    free(g_nnue.metadata_json);
    free(g_nnue.file_bytes);
    memset(&g_nnue, 0, sizeof(g_nnue));
}

const TceNnueFileHeader *tce_nnue_loader_header(void)
{
    return g_nnue.loaded ? &g_nnue.header : NULL;
}

const TceNnueTensorInfo *tce_nnue_loader_tensors(void)
{
    return g_nnue.loaded ? g_nnue.tensors : NULL;
}

const unsigned char *tce_nnue_loader_payload(void)
{
    return g_nnue.loaded ? g_nnue.file_bytes + g_nnue.payload_offset : NULL;
}

size_t tce_nnue_loader_payload_size(void)
{
    return g_nnue.loaded ? g_nnue.payload_size : 0;
}

const char *tce_nnue_loader_metadata_json(void)
{
    return g_nnue.loaded ? g_nnue.metadata_json : NULL;
}

int tce_nnue_loader_is_loaded(void)
{
    return g_nnue.loaded;
}
