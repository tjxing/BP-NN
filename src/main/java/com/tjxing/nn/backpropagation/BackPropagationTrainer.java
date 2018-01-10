package com.tjxing.nn.backpropagation;

import com.tjxing.nn.DataSet;
import com.tjxing.nn.DataSet.DataRecord;
import com.tjxing.nn.NeuralNetwork;
import com.tjxing.nn.utils.Assertion;
import com.tjxing.nn.utils.Vector;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.Iterator;

public class BackPropagationTrainer {

    private final static Log log = LogFactory.getLog(BackPropagationTrainer.class);

    private double sigma;
    private int maxIter;
    private double tol;

    public BackPropagationTrainer() {
        this(0.01, 10, 0.001);
    }

    public BackPropagationTrainer(double sigma, int maxIter, double tol) {
        this.sigma = sigma;
        this.maxIter = maxIter;
        this.tol = tol;
    }

    public void train(NeuralNetwork nn, DataSet data) {
        int iterCount = 0;
        double diff = Double.MAX_VALUE;
        double prevCost = 0.0;
        while(iterCount < maxIter) {
            double cost = 0.0;
            Iterator<DataRecord> records = data.iterator();
            int count = 0;
            while(records.hasNext()) {
                DataRecord record = records.next();
                Vector<Double> output = nn.forward(record);
                for(int i = 0; i < output.getSize(); ++i) {
                    double x = output.getData(i) - record.getLabel().getData(i);
                    cost += x * x;
                    nn.prepareUpdate(x, i);
                }
                ++count;
            }

            cost = cost / ( 2 * count );
            diff = Math.abs(prevCost - cost);

            if(log.isDebugEnabled()) {
                log.debug("Iteration " + iterCount + ": cost=" + cost + " , diff=" + diff);
            }

            if(diff < tol) {
                break;
            }
            nn.update(sigma / count);

            prevCost = cost;
            ++iterCount;
        }
    }

}
