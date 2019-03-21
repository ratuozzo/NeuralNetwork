import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class Neural {

    private int _seed;
    private double _learningRate;
    private int _batchSizeTraining;
    private int _batchSizeTesting;
    private int _nEpochs;

    private int _numInputs;
    private int _numOutputs;
    private int _numHiddenNodes;

    String _filenameTrain;
    String _filenameTest;

    DataSetIterator _trainIter;
    DataSetIterator _testIter;

    MultiLayerConfiguration _conf;

    MultiLayerNetwork _model;
    Evaluation _eval;

    public Neural() throws IOException, InterruptedException {

        _seed = 123;
        _learningRate = 0.01;
        _batchSizeTraining = 300;
        _batchSizeTesting = 900;
        _nEpochs = 20;

        _numInputs = 2;
        _numOutputs = 1;
        _numHiddenNodes = 30;

        _filenameTrain  = new ClassPathResource("Normal/training_raul.csv").getFile().getPath();
        _filenameTest  = new ClassPathResource("Normal/eval_raul.csv").getFile().getPath();

        System.out.println(_filenameTrain);

        readData();

        _conf = new NeuralNetConfiguration.Builder()
                .seed(_seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(_learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(_numInputs).nOut(_numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nIn(_numHiddenNodes).nOut(_numOutputs).build())
                .build();

        System.out.println(_conf.toJson());

         _model = new MultiLayerNetwork(_conf);

        _eval = new Evaluation(_numOutputs);

    }

    private void readData() throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(_filenameTrain)));
        _trainIter = new RecordReaderDataSetIterator(rr,_batchSizeTraining,0,1);

        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(_filenameTest)));
        _testIter = new RecordReaderDataSetIterator(rrTest,_batchSizeTesting,0,1);

    }

    public void train() {

        _model.init();
        _model.setListeners(new ScoreIterationListener(100));

        for ( int n = 0; n < _nEpochs; n++) {
            _model.fit( _trainIter );
        }
    }

    public void evaluate() {
        while(_testIter.hasNext()){
            DataSet t = _testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = _model.output(features,false);
            System.out.println("data: " + predicted);
            _eval.eval(labels, predicted);
        }

        System.out.println(_eval.stats());
        System.out.println(_eval.confusionToString());
        System.out.println(_eval.getLabelsList());
    }

}
