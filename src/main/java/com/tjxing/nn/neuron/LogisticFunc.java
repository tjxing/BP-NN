package com.tjxing.nn.neuron;

import com.tjxing.nn.utils.Assertion;
import com.tjxing.nn.utils.Vector;

public class LogisticFunc implements ActivationFunc {

    @Override
    public double apply(double x) {
        return sigmoid(x);
    }

    @Override
    public double diff(double x) {
        return x * (1 - x);
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

}
