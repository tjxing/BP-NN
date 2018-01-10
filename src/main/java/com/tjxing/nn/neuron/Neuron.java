package com.tjxing.nn.neuron;

import com.tjxing.nn.layer.Layer;
import com.tjxing.nn.utils.Vector;

import java.util.Random;

public class Neuron {

    private int inputSize = -1;
    private double[] weight = null;
    private double[] steps = null;
    private double inputWithWeight = 0.0;

    private ActivationFunc func = null;

    private Neuron() {

    }

    private boolean initialize() {
        weight = new double[inputSize];
        steps = new double[inputSize];
        Random random = new Random();
        for(int i = 0; i < weight.length; ++i) {
            weight[i] = random.nextInt(10) * 0.005;
            steps[i] = 0.0;
        }

        return true;
    }

    public double getOutput(Vector<Double> input) {
        for(int i = 0; i < inputSize; ++i) {
            inputWithWeight += input.getData(i) * weight[i];
        }
        return func.apply(inputWithWeight);
    }

    public void prepareUpdate(double diff, Layer prev) {
        double d = func.diff(inputWithWeight);
        for(int i = 0; i <steps.length; ++i) {
            steps[i] += diff * d * prev.getData(i);
            prev.prepareUpdate(diff * d * weight[i], i);
        }
    }

    public void update(double sigma) {
        for(int i = 0; i < weight.length; ++i) {
            weight[i] -= steps[i] * sigma;
            steps[i] = 0.0;
        }
        inputWithWeight = 0.0;
    }

    public static class Builder {

        Neuron neuron = new Neuron();

        public Builder setInputSize(int inputSize) {
            neuron.inputSize = inputSize;
            return this;
        }

        public Builder setActivationFunc(ActivationFunc func) {
            neuron.func = func;
            return this;
        }

        public Neuron build() {
            return neuron.initialize() ? neuron : null;
        }

    }

}
