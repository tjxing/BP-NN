package com.tjxing.nn.layer;

import com.tjxing.nn.utils.Vector;

public interface Layer extends Vector<Double> {

    void clean();

    void prepareUpdate(double diff, int index);

    void update(double sigma);

}
