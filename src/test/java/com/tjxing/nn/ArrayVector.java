package com.tjxing.nn;

import com.tjxing.nn.utils.Vector;

public class ArrayVector implements Vector<Double> {

    private final Double[] data;

    public ArrayVector(Double[] data) {
        this.data = data;
    }

    @Override
    public int getSize() {
        return data.length;
    }

    @Override
    public Double getData(int index) {
        return data[index];
    }
}
