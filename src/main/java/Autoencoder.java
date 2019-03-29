import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.RmsProp;

import java.io.IOException;
import java.util.*;

public class Autoencoder {

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

    private double _totalScore;
    private double _treshold;

    MultiLayerConfiguration _conf;

    MultiLayerNetwork _model;
    Evaluation _eval;




    public Autoencoder() throws IOException, InterruptedException {

        _totalScore = 0;
        _seed = 123;
        _learningRate = 0.01;
        _batchSizeTraining = 1;
        _batchSizeTesting = 1;
        _nEpochs = 10;

        _numInputs = 3;
        _numOutputs = 3;
        _numHiddenNodes = 30;


        _filenameTrain  = new ClassPathResource("OneClass/training_raul.csv").getFile().getPath();
        _filenameTest  = new ClassPathResource("OneClass/eval_raul.csv").getFile().getPath();

        _trainIter = new AnomalyDataSetIterator(new ClassPathResource("OneClass/training_raul.csv").getFile().getPath(), _batchSizeTraining);
        _testIter = new AnomalyDataSetIterator(new ClassPathResource("OneClass/eval_raul.csv").getFile().getPath(), _batchSizeTesting);

        initModel();

        _eval = new Evaluation(_numOutputs);

    }

    private void initModel(){
        _conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .list()
                .layer(0, new VariationalAutoencoder.Builder()
                        .activation(Activation.LEAKYRELU)
                        .encoderLayerSizes(50,50)        //2 encoder layers, each of size 256
                        .decoderLayerSizes(50, 50)        //2 decoder layers, each of size 256
                        .pzxActivationFunction(Activation.IDENTITY)  //p(z|data) activation function
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
                        .nIn(3)
                        .nOut(3)                        
                        .build())
                .pretrain(true).backprop(false).build();

        /*_conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(123456)
                .optimizationAlgo( OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp.Builder().learningRate(0.05).rmsDecay(0.002).build())
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(0, new AutoEncoder.Builder().name("input").nIn(_numInputs).nOut(2).build())
                .layer(1, new AutoEncoder.Builder().name("encoder1").nOut(1).build())
                .layer(2, new AutoEncoder.Builder().name("decoder1").nOut(2).build())
                .layer(3, new OutputLayer.Builder().name("output").nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();*/

        //System.out.println(_conf.toJson());



        _model = new MultiLayerNetwork(_conf);
        _model.init();

    }

    public void train() {
        for ( int n = 0; n < _nEpochs; n++) {
            _model.pretrain( _trainIter );
            System.out.println("Epoch: " +n);
        }

        calculateThreshold();
    }

    private void calculateThreshold() {

        int trainDatacount=0;
        _trainIter.reset();
        while(_trainIter.hasNext()){
            trainDatacount++;
            DataSet t = _trainIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = _model.output(features,false);

            double score = calculateScore(predicted);
            System.out.println("Score: "+score);
            _totalScore += score;
        }

        _treshold = _totalScore/trainDatacount;
        System.out.println("Threshold: "+_treshold);
    }

    private double calculateScore(INDArray predicted) {
        return  Math.abs(predicted.data().getFloat(0)) +
                Math.abs(predicted.data().getFloat(1)) +
                Math.abs(predicted.data().getFloat(2));
    }

    public void evaluate() {

        List<Pair<Double,Integer>> evalList = new ArrayList<Pair<Double,Integer>>();
        int aux = 0;
        while(_testIter.hasNext()){
            DataSet t = _testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = _model.output(features,false);


            double score = calculateScore(predicted);

            System.out.println("Label: "+labels+" Data: " + predicted+ " Score: "+score);



            evalList.add(new ImmutablePair<>(score, aux));
            aux++;
        }

        Collections.sort(evalList, Comparator.comparing(Pair::getRight));
        Stack<Integer> anomalyData = new Stack<>();

        System.out.println("Size: " + evalList.size());


        for (Pair<Double, Integer> pair: evalList) {
            double s = pair.getLeft();
            if (s >  _treshold) {
                anomalyData.push(pair.getRight());
            }
        }


        //output anomaly data
        System.out.println("based on the score, all anomaly data is following with descending order:\n");
        int anomalies = anomalyData.size();
        for (int i = anomalyData.size(); i > 0; i--) {
            System.out.println(anomalyData.pop());
        }

        System.out.println("Number of anomlies: " + anomalies);
    }

}
