package com.tjxing.nn.neuron;

import com.tjxing.nn.utils.Vector;

public interface ActivationFunc {

    double apply(double x);

    double diff(double x);

}
