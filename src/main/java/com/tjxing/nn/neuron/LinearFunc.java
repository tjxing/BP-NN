package com.tjxing.nn.neuron;

import com.tjxing.nn.utils.Assertion;
import com.tjxing.nn.utils.Vector;

public class LinearFunc implements ActivationFunc {

    // y = a * x + b
    private double a;
    private double b;

    public LinearFunc() {
        this(1.0, 0.0);
    }

    public LinearFunc(double a, double b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public double apply(double x) {
        return a * x + b;
    }

    @Override
    public double diff(double x) {
        return a;
    }

}
