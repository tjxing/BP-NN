package com.tjxing.nn;

import com.tjxing.nn.utils.Vector;

import java.io.IOException;
import java.io.InputStream;

public abstract class Reader {

    public abstract int getCount();

    public abstract Vector<Double> read();

    protected int readInt(InputStream stream) throws IOException {
        int result = 0;
        for(int i = 0; i < 4; ++i) {
            result = result << 8 | stream.read();
        }
        return result;
    }

}
