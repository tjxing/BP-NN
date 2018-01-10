package com.tjxing.nn;

import com.tjxing.nn.utils.Vector;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class ImageReader extends Reader implements Closeable {

    private final static Log log = LogFactory.getLog(LabelReader.class);

    private File file = null;
    private boolean binary = true;

    private FileInputStream stream = null;
    private int count = 0;
    private int row = 0;
    private int col = 0;
    private int recordSize = 0;

    public ImageReader(File file) {
        this(file, true);
    }

    public ImageReader(File file, boolean binary) {
        this.file = file;
        this.binary = binary;

        try {
            stream = new FileInputStream(file);
            int msb = readInt(stream);
            log.debug("msb: " + msb);
            count = readInt(stream);
            log.info("Record count: " + count);
            row = readInt(stream);
            col = readInt(stream);
            log.info("Image size: " + row + "*" + col);
            recordSize = row * col;
        } catch (IOException e) {
            log.error("Fail to read image file", e);
        }
    }

    @Override
    public int getCount() {
        return count;
    }

    @Override
    public Vector<Double> read() {
        byte[] bytes = new byte[recordSize];
        try {
            if(stream.read(bytes) < recordSize) {
                throw new IOException("Error");
            }
            return new Vector<Double>() {
                @Override
                public int getSize() {
                    return recordSize;
                }

                @Override
                public Double getData(int index) {
                    int x = bytes[index] & 0xff;
                    if(binary) {
                        return x >= 128 ? 1.0 : 0.0;
                    } else {
                        return Integer.valueOf(x).doubleValue();
                    }
                }
            };
        } catch (IOException e) {
            log.error("Fail to read image file", e);
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
