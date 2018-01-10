package com.tjxing.nn;

import com.tjxing.nn.utils.Vector;

public interface DataSet extends Iterable<DataSet.DataRecord> {

    public abstract Integer getDataSizeHint();

    public interface DataRecord {

        Vector<Double> getInput();

        Vector<Double> getLabel();

    }

}
