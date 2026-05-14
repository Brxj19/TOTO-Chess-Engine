#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tce_nnue/tce_nnue.h"

#ifndef TCE_NNUE_INFER_TOLERANCE
#define TCE_NNUE_INFER_TOLERANCE 5
#endif

typedef struct {
    int index;
    int side_to_move;
    int target_cp;
    int expected_pred_cp;
    int *white_features;
    int white_count;
    int *black_features;
    int black_count;
} Vector;

static char *read_text_file(const char *path)
{
    FILE *file = fopen(path, "rb");
    long size;
    char *buffer;

    if (!file) {
        fprintf(stderr, "failed to open %s\n", path);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    size = ftell(file);
    rewind(file);
    if (size < 0) {
        fclose(file);
        return NULL;
    }
    buffer = (char *)malloc((size_t)size + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    buffer[size] = '\0';
    fclose(file);
    return buffer;
}

static const char *skip_ws(const char *p)
{
    while (*p && isspace((unsigned char)*p))
        p++;
    return p;
}

static int parse_int_field(const char *object, const char *key, int *out)
{
    char pattern[64];
    const char *p;
    char *end;
    long value;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    p = strstr(object, pattern);
    if (!p)
        return 1;
    p = strchr(p + strlen(pattern), ':');
    if (!p)
        return 1;
    p = skip_ws(p + 1);
    value = strtol(p, &end, 10);
    if (end == p)
        return 1;
    *out = (int)value;
    return 0;
}

static int parse_int_array(const char *object, const char *key, int **out, int *count)
{
    char pattern[64];
    const char *p;
    int capacity = 32;
    int used = 0;
    int *values = (int *)malloc((size_t)capacity * sizeof(values[0]));

    if (!values)
        return 1;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    p = strstr(object, pattern);
    if (!p)
        goto fail;
    p = strchr(p + strlen(pattern), '[');
    if (!p)
        goto fail;
    p++;

    for (;;) {
        char *end;
        long value;
        p = skip_ws(p);
        if (*p == ']')
            break;
        value = strtol(p, &end, 10);
        if (end == p)
            goto fail;
        if (used == capacity) {
            int new_capacity = capacity * 2;
            int *new_values = (int *)realloc(values, (size_t)new_capacity * sizeof(values[0]));
            if (!new_values)
                goto fail;
            values = new_values;
            capacity = new_capacity;
        }
        values[used++] = (int)value;
        p = skip_ws(end);
        if (*p == ',')
            p++;
        else if (*p != ']')
            goto fail;
    }

    *out = values;
    *count = used;
    return 0;

fail:
    free(values);
    return 1;
}

static void free_vector(Vector *vector)
{
    free(vector->white_features);
    free(vector->black_features);
    memset(vector, 0, sizeof(*vector));
}

static int parse_vector_object(char *object, Vector *vector)
{
    memset(vector, 0, sizeof(*vector));
    if (parse_int_field(object, "index", &vector->index) ||
        parse_int_field(object, "side_to_move", &vector->side_to_move) ||
        parse_int_field(object, "target_cp", &vector->target_cp) ||
        parse_int_field(object, "expected_pred_cp", &vector->expected_pred_cp) ||
        parse_int_array(object, "white_features", &vector->white_features, &vector->white_count) ||
        parse_int_array(object, "black_features", &vector->black_features, &vector->black_count)) {
        free_vector(vector);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    char *json;
    char *p;
    int failures = 0;
    int checked = 0;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <network.tcennue> <vectors.json>\n", argv[0]);
        return 2;
    }

    if (tce_nnue_load(argv[1]) != 0) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    json = read_text_file(argv[2]);
    if (!json) {
        tce_nnue_free();
        return 1;
    }

    p = json;
    while ((p = strchr(p, '{')) != NULL) {
        char *end = strchr(p, '}');
        char saved;
        Vector vector;
        int actual_cp;
        int diff;

        if (!end)
            break;
        saved = end[1];
        end[1] = '\0';

        if (parse_vector_object(p, &vector)) {
            fprintf(stderr, "failed to parse vector near %.40s\n", p);
            end[1] = saved;
            failures++;
            break;
        }

        if (tce_nnue_evaluate_sparse(
                vector.white_features,
                vector.white_count,
                vector.black_features,
                vector.black_count,
                vector.side_to_move,
                &actual_cp) != 0) {
            fprintf(stderr, "inference failed for vector %d\n", vector.index);
            failures++;
        } else {
            diff = actual_cp - vector.expected_pred_cp;
            printf("%d %d %d %d\n",
                   vector.index,
                   vector.expected_pred_cp,
                   actual_cp,
                   diff);
            if (abs(diff) > TCE_NNUE_INFER_TOLERANCE)
                failures++;
            checked++;
        }

        free_vector(&vector);
        end[1] = saved;
        p = end + 1;
    }

    free(json);
    tce_nnue_free();

    if (failures) {
        fprintf(stderr, "checked %d vectors with %d failures\n", checked, failures);
        return 1;
    }

    printf("checked %d vectors successfully\n", checked);
    return 0;
}
