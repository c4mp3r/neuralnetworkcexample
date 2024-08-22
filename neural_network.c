#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_NEURONS 5
#define HIDDEN_NEURONS 3
#define OUTPUT_NEURONS 5
#define MAX_LINE_LENGTH 1024

// to compile > gcc -o neural_network neural_network.c -lm
// Sigmoid function from math.h
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// NN structure
typedef struct {
    double input[INPUT_NEURONS];
    double hidden[HIDDEN_NEURONS];
    double output[OUTPUT_NEURONS];
    double weight_input_hidden[INPUT_NEURONS][HIDDEN_NEURONS];
    double weight_hidden_output[HIDDEN_NEURONS][OUTPUT_NEURONS];
    double bias_hidden[HIDDEN_NEURONS];
    double bias_output[OUTPUT_NEURONS];
} NeuralNetwork;

// Initialize the neural network with random weights
void initialize(NeuralNetwork *nn) {
    for (int i = 0; i < INPUT_NEURONS; i++) {
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            nn->weight_input_hidden[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        for (int j = 0; j < OUTPUT_NEURONS; j++) {
            nn->weight_hidden_output[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        nn->bias_hidden[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        nn->bias_output[i] = ((double)rand() / RAND_MAX) - 0.5;
    }
}

// Feedforward function to calculate the output
void feedforward(NeuralNetwork *nn) {
    // Calculate the hidden layer
    for (int i = 0; i < HIDDEN_NEURONS; i++) {
        nn->hidden[i] = 0;
        for (int j = 0; j < INPUT_NEURONS; j++) {
            nn->hidden[i] += nn->input[j] * nn->weight_input_hidden[j][i];
        }
        nn->hidden[i] += nn->bias_hidden[i];
        nn->hidden[i] = sigmoid(nn->hidden[i]);
    }

    // Calculate the output layer
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        nn->output[i] = 0;
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            nn->output[i] += nn->hidden[j] * nn->weight_hidden_output[j][i];
        }
        nn->output[i] += nn->bias_output[i];
        nn->output[i] = sigmoid(nn->output[i]);
    }
}

// Input from a CSV file
void read_csv_and_feedforward(NeuralNetwork *nn, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        // Tokenize the line and fill the input array
        char *token = strtok(line, ",");
        for (int i = 0; i < INPUT_NEURONS && token != NULL; i++) {
            nn->input[i] = atof(token);
            token = strtok(NULL, ",");
        }

        // Perform feedforward
        feedforward(nn);

        // Print the output
        printf("Output: ");
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            printf("%f ", nn->output[i]);
        }
        printf("\n");
    }

    fclose(file);
}

int main() {
    NeuralNetwork nn;
    initialize(&nn);

    const char *filename = "input_data.csv";
    read_csv_and_feedforward(&nn, filename);

    return 0;
}

