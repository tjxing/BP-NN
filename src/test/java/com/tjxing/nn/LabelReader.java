package com.tjxing.nn;

import com.tjxing.nn.utils.Vector;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.*;

public class LabelReader extends Reader implements Closeable {

    private final static Log log = LogFactory.getLog(LabelReader.class);

    private File file = null;
    private boolean encoded = true;

    private FileInputStream stream = null;
    private int count = 0;

    public LabelReader(File file) {
        this(file, true);
    }

    public LabelReader(File file, boolean encoded) {
        this.file = file;
        this.encoded = encoded;

        try {
            stream = new FileInputStream(file);
            int msb = readInt(stream);
            log.debug("msb: " + msb);
            count = readInt(stream);
            log.info("Record count: " + count);
        } catch (IOException e) {
            log.error("Fail to read label file", e);
        }
    }

    @Override
    public int getCount() {
        return count;
    }

    @Override
    public Vector<Double> read() {
        try {
            int value = stream.read();
            if(encoded) {
                return new Vector<Double>() {
                    private final int x = value;

                    @Override
                    public int getSize() {
                        return 10;
                    }

                    @Override
                    public Double getData(int index) {
                        return index == x ? 1.0 : 0.0;
                    }
                };
            } else {
                return new ArrayVector(new Double[] {Integer.valueOf(value).doubleValue()});
            }
        } catch (IOException e) {
            log.error("Fail to read label file", e);
        }
        return null;
    }

    @Override
    public void close() throws IOException {
        if(stream != null) {
            stream.close();
        }
    }

}
