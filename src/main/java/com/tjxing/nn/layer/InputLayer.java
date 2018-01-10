package com.tjxing.nn.layer;

import com.tjxing.nn.DataSet.DataRecord;
import com.tjxing.nn.utils.Assertion;
import com.tjxing.nn.utils.Vector;

public class InputLayer implements Layer {

    private int size = -1;
    private Vector<Double> input = null;

    public InputLayer(int size) {
        this.size = size;
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    public Double getData(int index) {
        return input.getData(index);
    }

    public void clean() {
        input = null;
    }

    public void setInput(DataRecord record) {
        Assertion.assertEqual(record.getInput().getSize(), size);

        this.input = record.getInput();
    }

    public void setInput(Vector<Double> input) {
        Assertion.assertEqual(input.getSize(), size);

        this.input = input;
    }

    @Override
    public void prepareUpdate(double diff, int index) {
        // Do nothing
    }

    @Override
    public void update(double sigma) {
        // Do nothing
    }
}