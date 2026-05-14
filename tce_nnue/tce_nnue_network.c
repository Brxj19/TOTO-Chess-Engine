#include "tce_nnue_network.h"

#include "tce_nnue_accumulator.h"
#include "tce_nnue_features.h"
#include "tce_nnue_loader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double ft_weight;
    double hidden1_weight;
    double hidden1_bias;
    double hidden2_weight;
    double hidden2_bias;
    double output_weight;
    double output_bias;
} TceNnueScales;

static double clipped_relu(double value)
{
    if (value < 0.0)
        return 0.0;
    if (value > 1.0)
        return 1.0;
    return value;
}

static int parse_json_double(const char *json, const char *key, double *out)
{
    char pattern[64];
    const char *p;
    char *end;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    p = strstr(json, "\"weight_scales\"");
    if (!p)
        return 1;
    p = strstr(p, pattern);
    if (!p)
        return 1;
    p = strchr(p + strlen(pattern), ':');
    if (!p)
        return 1;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        p++;
    *out = strtod(p, &end);
    if (end == p)
        return 1;
    return 0;
}

static int load_scales(TceNnueScales *scales)
{
    const char *json = tce_nnue_loader_metadata_json();
    if (!json)
        return 1;

    if (parse_json_double(json, "ft_weight", &scales->ft_weight) ||
        parse_json_double(json, "hidden1_weight", &scales->hidden1_weight) ||
        parse_json_double(json, "hidden1_bias", &scales->hidden1_bias) ||
        parse_json_double(json, "hidden2_weight", &scales->hidden2_weight) ||
        parse_json_double(json, "hidden2_bias", &scales->hidden2_bias) ||
        parse_json_double(json, "output_weight", &scales->output_weight) ||
        parse_json_double(json, "output_bias", &scales->output_bias)) {
        fprintf(stderr, "tce_nnue: failed to parse quantization scales\n");
        return 1;
    }

    return 0;
}

static const void *tensor_ptr(int tensor_index)
{
    const unsigned char *payload = tce_nnue_loader_payload();
    const TceNnueTensorInfo *tensors = tce_nnue_loader_tensors();
    if (!payload || !tensors)
        return NULL;
    return payload + tensors[tensor_index].offset;
}

static int validate_shapes(const TceNnueFileHeader *header,
                           const TceNnueTensorInfo *tensors)
{
    if (header->output_dim != 1) {
        fprintf(stderr, "tce_nnue: only output_dim=1 is supported\n");
        return 1;
    }

    if (tensors[TCE_NNUE_TENSOR_FT_WEIGHT].rank != 2 ||
        tensors[TCE_NNUE_TENSOR_FT_WEIGHT].shape[0] != header->feature_count ||
        tensors[TCE_NNUE_TENSOR_FT_WEIGHT].shape[1] != header->half_dim)
        return 1;

    if (tensors[TCE_NNUE_TENSOR_HIDDEN1_WEIGHT].rank != 2 ||
        tensors[TCE_NNUE_TENSOR_HIDDEN1_WEIGHT].shape[0] != header->hidden1_dim ||
        tensors[TCE_NNUE_TENSOR_HIDDEN1_WEIGHT].shape[1] != header->half_dim * 2)
        return 1;

    if (tensors[TCE_NNUE_TENSOR_HIDDEN1_BIAS].rank != 1 ||
        tensors[TCE_NNUE_TENSOR_HIDDEN1_BIAS].shape[0] != header->hidden1_dim)
        return 1;

    if (tensors[TCE_NNUE_TENSOR_HIDDEN2_WEIGHT].rank != 2 ||
        tensors[TCE_NNUE_TENSOR_HIDDEN2_WEIGHT].shape[0] != header->hidden2_dim ||
        tensors[TCE_NNUE_TENSOR_HIDDEN2_WEIGHT].shape[1] != header->hidden1_dim)
        return 1;

    if (tensors[TCE_NNUE_TENSOR_HIDDEN2_BIAS].rank != 1 ||
        tensors[TCE_NNUE_TENSOR_HIDDEN2_BIAS].shape[0] != header->hidden2_dim)
        return 1;

    if (tensors[TCE_NNUE_TENSOR_OUTPUT_WEIGHT].rank != 2 ||
        tensors[TCE_NNUE_TENSOR_OUTPUT_WEIGHT].shape[0] != 1 ||
        tensors[TCE_NNUE_TENSOR_OUTPUT_WEIGHT].shape[1] != header->hidden2_dim)
        return 1;

    if (tensors[TCE_NNUE_TENSOR_OUTPUT_BIAS].rank != 1 ||
        tensors[TCE_NNUE_TENSOR_OUTPUT_BIAS].shape[0] != 1)
        return 1;

    return 0;
}

static double dense_from_int_acc(
    const int16_t *weights,
    const int32_t *bias,
    int row,
    const int32_t *stm_acc,
    const int32_t *opp_acc,
    int half_dim,
    const TceNnueScales *scales
)
{
    int64_t sum = bias[row];
    const int16_t *wrow = weights + ((size_t)row * (size_t)half_dim * 2u);

    for (int i = 0; i < half_dim; i++)
        sum += (int64_t)wrow[i] * stm_acc[i];
    for (int i = 0; i < half_dim; i++)
        sum += (int64_t)wrow[half_dim + i] * opp_acc[i];

    return (double)sum * scales->hidden1_bias;
}

static double dense_from_double(
    const int16_t *weights,
    const int32_t *bias,
    int row,
    const double *input,
    int input_dim,
    double weight_scale,
    double bias_scale
)
{
    double sum = (double)bias[row] * bias_scale;
    const int16_t *wrow = weights + ((size_t)row * (size_t)input_dim);

    for (int i = 0; i < input_dim; i++)
        sum += ((double)wrow[i] * weight_scale) * input[i];

    return sum;
}

int tce_nnue_evaluate_sparse(
    const int *white_features,
    int white_count,
    const int *black_features,
    int black_count,
    int side_to_move,
    int *out_cp
)
{
    const TceNnueFileHeader *header = tce_nnue_loader_header();
    const TceNnueTensorInfo *tensors = tce_nnue_loader_tensors();
    const int16_t *ft_weight;
    const int16_t *hidden1_weight;
    const int32_t *hidden1_bias;
    const int16_t *hidden2_weight;
    const int32_t *hidden2_bias;
    const int16_t *output_weight;
    const int32_t *output_bias;
    TceNnueScales scales;
    int32_t *white_acc = NULL;
    int32_t *black_acc = NULL;
    double *hidden1 = NULL;
    double *hidden2 = NULL;
    const int32_t *stm_acc;
    const int32_t *opp_acc;
    double output;
    int result = 1;

    if (!out_cp) {
        fprintf(stderr, "tce_nnue: null output pointer\n");
        return 1;
    }

    if (!tce_nnue_loader_is_loaded() || !header || !tensors) {
        fprintf(stderr, "tce_nnue: no .tcennue file loaded\n");
        return 1;
    }

    if (side_to_move != 0 && side_to_move != 1) {
        fprintf(stderr, "tce_nnue: side_to_move must be 0 or 1\n");
        return 1;
    }

    if (validate_shapes(header, tensors)) {
        fprintf(stderr, "tce_nnue: tensor shapes do not match header\n");
        return 1;
    }

    if (tce_nnue_validate_features(white_features, white_count, (int)header->feature_count) ||
        tce_nnue_validate_features(black_features, black_count, (int)header->feature_count) ||
        load_scales(&scales))
        return 1;

    ft_weight = (const int16_t *)tensor_ptr(TCE_NNUE_TENSOR_FT_WEIGHT);
    hidden1_weight = (const int16_t *)tensor_ptr(TCE_NNUE_TENSOR_HIDDEN1_WEIGHT);
    hidden1_bias = (const int32_t *)tensor_ptr(TCE_NNUE_TENSOR_HIDDEN1_BIAS);
    hidden2_weight = (const int16_t *)tensor_ptr(TCE_NNUE_TENSOR_HIDDEN2_WEIGHT);
    hidden2_bias = (const int32_t *)tensor_ptr(TCE_NNUE_TENSOR_HIDDEN2_BIAS);
    output_weight = (const int16_t *)tensor_ptr(TCE_NNUE_TENSOR_OUTPUT_WEIGHT);
    output_bias = (const int32_t *)tensor_ptr(TCE_NNUE_TENSOR_OUTPUT_BIAS);

    if (!ft_weight || !hidden1_weight || !hidden1_bias || !hidden2_weight ||
        !hidden2_bias || !output_weight || !output_bias)
        return 1;

    white_acc = (int32_t *)malloc((size_t)header->half_dim * sizeof(*white_acc));
    black_acc = (int32_t *)malloc((size_t)header->half_dim * sizeof(*black_acc));
    hidden1 = (double *)malloc((size_t)header->hidden1_dim * sizeof(*hidden1));
    hidden2 = (double *)malloc((size_t)header->hidden2_dim * sizeof(*hidden2));
    if (!white_acc || !black_acc || !hidden1 || !hidden2) {
        fprintf(stderr, "tce_nnue: out of memory during inference\n");
        goto done;
    }

    if (tce_nnue_refresh_accumulator(ft_weight, (int)header->half_dim,
                                     white_features, white_count, white_acc) ||
        tce_nnue_refresh_accumulator(ft_weight, (int)header->half_dim,
                                     black_features, black_count, black_acc))
        goto done;

    stm_acc = side_to_move == 0 ? white_acc : black_acc;
    opp_acc = side_to_move == 0 ? black_acc : white_acc;

    for (uint32_t row = 0; row < header->hidden1_dim; row++) {
        hidden1[row] = clipped_relu(dense_from_int_acc(
            hidden1_weight, hidden1_bias, (int)row, stm_acc, opp_acc,
            (int)header->half_dim, &scales));
    }

    for (uint32_t row = 0; row < header->hidden2_dim; row++) {
        hidden2[row] = clipped_relu(dense_from_double(
            hidden2_weight, hidden2_bias, (int)row, hidden1,
            (int)header->hidden1_dim, scales.hidden2_weight,
            scales.hidden2_bias));
    }

    output = dense_from_double(
        output_weight, output_bias, 0, hidden2, (int)header->hidden2_dim,
        scales.output_weight, scales.output_bias);

    *out_cp = (int)llround(output * header->target_scale);
    result = 0;

done:
    free(white_acc);
    free(black_acc);
    free(hidden1);
    free(hidden2);
    return result;
}
