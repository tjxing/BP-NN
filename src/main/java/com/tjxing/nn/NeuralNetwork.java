package com.tjxing.nn;

import com.tjxing.nn.DataSet.DataRecord;
import com.tjxing.nn.backpropagation.BackPropagationTrainer;
import com.tjxing.nn.layer.InputLayer;
import com.tjxing.nn.layer.Layer;
import com.tjxing.nn.layer.OutputLayer;
import com.tjxing.nn.utils.Vector;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class NeuralNetwork {

    private static final Log log = LogFactory.getLog(NeuralNetwork.class);

    private int inputSize = 0;
    private int outputSize = 0;
    private List<Integer> hiddenLayersSize = null;
    private boolean computeWithLogistic = true;
    private boolean outputWithSoftmax = false;

    private InputLayer inputLayer = null;
    private OutputLayer outputLayer = null;
    private double outputSum = 0.0;

    private NeuralNetwork() {

    }

    private boolean initialize() {
        if(inputSize <= 0 || outputSize <= 0) {
            return false;
        }
        inputLayer = new InputLayer(inputSize);

        Layer layer = inputLayer;
        if(hiddenLayersSize != null && hiddenLayersSize.size() > 0) {
            for(int size : hiddenLayersSize) {
                OutputLayer hidden = new OutputLayer(size, layer);
                layer = hidden;
            }
        }
        outputLayer = new OutputLayer(outputSize, layer);

        return true;
    }

    public Vector<Double> forward(Vector<Double> input) {
        clean();
        inputLayer.setInput(input);
        if(outputWithSoftmax) {
            for(int i = 0; i < outputLayer.getSize(); ++i) {
                outputSum += outputLayer.getData(i);
            }
            return new Vector<Double>() {
                @Override
                public int getSize() {
                    return outputSize;
                }

                @Override
                public Double getData(int index) {
                    return outputLayer.getData(index) / outputSum;
                }
            };
        }
        return outputLayer;
    }

    public Vector<Double> forward(DataRecord input) {
        return forward(input.getInput());
    }

    private void clean() {
        outputSum = 0.0;
        outputLayer.clean();
    }

    public void train(DataSet data) {
        BackPropagationTrainer trainer = new BackPropagationTrainer();
        trainer.train(this, data);
    }

    public void prepareUpdate(double diff, int index) {
        if(outputWithSoftmax) {
            diff *= (1 / outputSum - outputLayer.getData(index) / outputSum / outputSum);
        }
        outputLayer.prepareUpdate(diff, index);
    }

    public void update(double sigma) {
        outputLayer.update(sigma);
    }

    public static class Builder {

        private NeuralNetwork nn = null;

        public Builder() {
            nn = new NeuralNetwork();
        }

        public Builder(int inputSize, int outputSize) {
            this();
            nn.inputSize = inputSize;
            nn.outputSize = outputSize;
        }

        public Builder setInputSize(int inputSize) {
            nn.inputSize = inputSize;
            return this;
        }

        public Builder setOutputSize(int outputSize) {
            nn.outputSize = outputSize;
            return this;
        }

        public Builder addHiddenLayer(int...layerSize) {
            if(nn.hiddenLayersSize == null) {
                nn.hiddenLayersSize = new ArrayList<Integer>();
            }

            for(int size : layerSize) {
                nn.hiddenLayersSize.add(size);
            }
            return this;
        }

        public Builder setComputeWithLogistic(boolean computeWithLogistic) {
            //TODO: Implement computeWithLogistic=false later
            return this;
        }

        public Builder setOutputWithSoftmax(boolean outputWithSoftmax) {
            nn.outputWithSoftmax = outputWithSoftmax;
            return this;
        }

        public NeuralNetwork build() {
            return nn.initialize() ? nn : null;
        }

    }

}