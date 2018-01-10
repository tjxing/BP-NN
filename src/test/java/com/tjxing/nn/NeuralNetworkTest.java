package com.tjxing.nn;

import com.tjxing.nn.DataSet.DataRecord;
import com.tjxing.nn.utils.Assertion;
import com.tjxing.nn.utils.Vector;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkTest {

    @BeforeEach
    void setUp() {
    }

    @AfterEach
    void tearDown() {
     }

    @Test
    public void forwardPropagation() {
        Vector<Double>[] inVec = new ArrayVector[] { new ArrayVector(new Double[] {1.0, 2.0, 3.0, 4.0, 5.0}), new ArrayVector(new Double[] {0.1, 0.2, 0.3, 0.4, 0.5}) };
        Vector<Double>[] labelVec = new ArrayVector[] { new ArrayVector(new Double[] {1.0, 0.0}), new ArrayVector(new Double[] {0.0, 1.0}) };

        DataSet data = new DataSet() {
            @Override
            public Iterator<DataRecord> iterator() {
                return new Iterator<DataRecord>() {
                    private int index = 0;

                    @Override
                    public boolean hasNext() {
                        return index < inVec.length;
                    }

                    @Override
                    public DataRecord next() {
                        DataRecord record = new DataRecord() {
                            private final int i = index;

                            @Override
                            public Vector<Double> getInput() {
                                return inVec[i];
                            }

                            @Override
                            public Vector<Double> getLabel() {
                                return labelVec[i];
                            }
                        };
                        ++index;
                        return record;
                    }
                };
            }

            @Override
            public Integer getDataSizeHint() {
                return 2;
            }
        };

        NeuralNetwork nn = new NeuralNetwork.Builder().setInputSize(5).addHiddenLayer(4, 3).setOutputSize(2).build();
        nn.train(data);

//        int count = 1;
//        Iterator<DataRecord> records = data.iterator();
//        while(records.hasNext()) {
//            Vector<Double> record = records.next().getInput();
//            Vector<Double> output = nn.forward(record);
//
//            System.out.println("Iteration " + count++);
//            System.out.print("input: [");
//            for(int i = 0; i < record.getSize(); ++i) {
//                if(i > 0) {
//                    System.out.print(", ");
//                }
//                System.out.print(record.getData(i));
//            }
//            System.out.println("]");
//
//            System.out.print("output: [");
//            for(int i = 0; i < output.getSize(); ++i) {
//                if(i > 0) {
//                    System.out.print(", ");
//                }
//                System.out.print(output.getData(i));
//            }
//            System.out.println("]\n");
//        }
    }

    @Test
    public void train() {
        DataSet trainDataset = loadDataset("train");
        //DataSet testDataset = loadDataset("t10k");

        NeuralNetwork nn = new NeuralNetwork.Builder()
                .setInputSize(28 * 28)
                .addHiddenLayer(100)
                .setOutputSize(10)
                .setOutputWithSoftmax(true)
                .build();
        nn.train(trainDataset);
    }

    private DataSet loadDataset(String name) {
        String path = "src/test/resources/data/";
        File imageFile = new File(path + name + "-images.idx3-ubyte");
        File labelFile = new File(path + name + "-labels.idx1-ubyte");
        if(!imageFile.exists() || !labelFile.exists()) {
            return null;
        }

        return new DataSet() {
            @Override
            public Integer getDataSizeHint() {
                return null;
            }

            @Override
            public Iterator<DataRecord> iterator() {
                return new Iterator<DataRecord>() {
                    private ImageReader imageReader = new ImageReader(imageFile);
                    private LabelReader labelReader = new LabelReader(labelFile);
                    {
                        Assertion.assertEqual(imageReader.getCount(), labelReader.getCount());
                    }

                    private int index = 0;

                    @Override
                    public boolean hasNext() {
                        boolean hasnext = index < 2;//imageReader.getCount();
                        if(!hasnext) {
                            try {
                                imageReader.close();
                                labelReader.close();
                            } catch (IOException e) {

                            }
                        }
                        return hasnext;
                    }

                    @Override
                    public DataRecord next() {
                        ++index;
                        return new DataRecord() {
                            @Override
                            public Vector<Double> getInput() {
                                return imageReader.read();
                            }

                            @Override
                            public Vector<Double> getLabel() {
                                return labelReader.read();
                            }
                        };
                    }
                };
            }
        };
    }

}