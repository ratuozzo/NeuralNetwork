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

    MultiLayerConfiguration _conf;

    MultiLayerNetwork _model;
    Evaluation _eval;


    org.deeplearning4j.nn.layers.variational.VariationalAutoencoder _vae;

    public Autoencoder() throws IOException, InterruptedException {

        _totalScore = 0;
        _seed = 123;
        _learningRate = 0.01;
        _batchSizeTraining = 30;
        _batchSizeTesting = 1;
        _nEpochs = 10000;

        _numInputs = 3;
        _numOutputs = 3;
        _numHiddenNodes = 30;


        _filenameTrain  = new ClassPathResource("OneClass/training_raul.csv").getFile().getPath();
        _filenameTest  = new ClassPathResource("OneClass/eval_raul.csv").getFile().getPath();

        _trainIter = new AnomalyDataSetIterator(new ClassPathResource("OneClass/training_raul.csv").getFile().getPath(), _batchSizeTraining);
        _testIter = new AnomalyDataSetIterator(new ClassPathResource("OneClass/eval_raul.csv").getFile().getPath(), _batchSizeTesting);


        _conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .list()
                .layer(0, new VariationalAutoencoder.Builder()
                        .activation(Activation.LEAKYRELU)
                        .encoderLayerSizes(3, 2)        //2 encoder layers, each of size 256
                        .decoderLayerSizes(2, 3)        //2 decoder layers, each of size 256
                        .pzxActivationFunction(Activation.IDENTITY)  //p(z|data) activation function
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
                        .nIn(3)                       //Input size: 28x28
                        .nOut(3)                            //Size of the latent variable space: p(z|x). 2 dimensions here for plotting, use more in general
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

        _eval = new Evaluation(_numOutputs);

    }

    public void train() {

        _model.init();
        _vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) _model.getLayer(0);

        _model.setListeners(new ScoreIterationListener(100));

        DataSet dSet = _trainIter.next();

        for ( int n = 0; n < _nEpochs; n++) {
            _model.fit( _trainIter );
        }
    }

    public void evaluate() {



        List<Pair<Double,Integer>> evalList = new ArrayList<Pair<Double,Integer>>();
        int aux = 0;
        while(_testIter.hasNext()){
            DataSet t = _testIter.next();
            System.out.println(_vae.score());
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = _model.output(features,false);
            System.out.println("data: " + predicted);
            System.out.println("Labels: " + labels);
            double score = Math.abs(Math.abs(predicted.data().getFloat(0)) - Math.abs(labels.data().getFloat(0))) +
            Math.abs(Math.abs(predicted.data().getFloat(1)) - Math.abs(labels.data().getFloat(1))) +
            Math.abs(Math.abs(predicted.data().getFloat(2)) - Math.abs(labels.data().getFloat(2)));
            _totalScore += score;
            evalList.add(new ImmutablePair<>(score, aux));
            aux++;
        }

        Collections.sort(evalList, Comparator.comparing(Pair::getRight));
        Stack<Integer> anomalyData = new Stack<>();
        System.out.println("Size: " + evalList.size());
        double threshold = (_totalScore / evalList.size());
        System.out.println("Threshold: " + threshold);
        for (Pair<Double, Integer> pair: evalList) {
            double s = pair.getLeft();
            if (s >  threshold) {
                anomalyData.push(pair.getRight());
            }
        }

        System.out.println(_eval.stats());
        System.out.println(_eval.confusionToString());
        System.out.println(_eval.getLabelsList());
        System.out.println("Score: " + _totalScore);

        //output anomaly data
        System.out.println("based on the score, all anomaly data is following with descending order:\n");
        int anomalies = anomalyData.size();
        for (int i = anomalyData.size(); i > 0; i--) {
            System.out.println(anomalyData.pop());
        }

        System.out.println("Number of anomlies: " + anomalies);
    }

}
