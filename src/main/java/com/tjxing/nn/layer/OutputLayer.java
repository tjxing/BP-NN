package com.tjxing.nn.layer;

import com.tjxing.nn.neuron.LinearFunc;
import com.tjxing.nn.neuron.LogisticFunc;
import com.tjxing.nn.neuron.Neuron;

public class OutputLayer implements Layer {

    private int outputSize = -1;
    private Layer prev = null;
    private boolean computeWithLogistic = true;
    private boolean outputWithSoftmax = false;

    private boolean init = false;

    private Neuron[] neurons = null;
    private Double[] cache;

    public OutputLayer(int outputSize, Layer prev) {
        this.outputSize = outputSize;
        this.prev = prev;
    }

    public OutputLayer(int outputSize, Layer prev, boolean computeWithLogistic, boolean outputWithSoftmax) {
        this(outputSize, prev);
        this.computeWithLogistic = computeWithLogistic;
        this.outputWithSoftmax = outputWithSoftmax;
    }

    private void initialize() {
        neurons = new Neuron[outputSize];
        for(int i = 0; i < outputSize; ++i) {
            Neuron neuron = new Neuron.Builder()
                    .setInputSize(prev.getSize())
                    .setActivationFunc(computeWithLogistic ? new LogisticFunc() : new LinearFunc())
                    .build();
            neurons[i] = neuron;
        }
        cache = new Double[outputSize];
        init = true;
    }

    @Override
    public int getSize() {
        return outputSize;
    }

    @Override
    public Double getData(int index) {
        if(!init) {
            initialize();
        }

        double result = 0.0;
        if(cache[index] != null) {
            result = cache[index];
        } else {
            result = neurons[index].getOutput(prev);
            cache[index] = result;
        }

        return result;
    }

    public void clean() {
        cache = new Double[outputSize];
        prev.clean();
    }

    @Override
    public void prepareUpdate(double diff, int index) {
        neurons[index].prepareUpdate(diff, prev);
    }

    @Override
    public void update(double sigma) {
        for(Neuron neuron : neurons) {
            neuron.update(sigma);
        }
        prev.update(sigma);
    }

}
