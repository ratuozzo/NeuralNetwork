import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.RmsProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A simple example of training a variational autoencoder on MNIST.
 * This example intentionally has a small hidden state Z (2 values) for visualization on a 2-grid.
 *
 * After training, this example plots 2 things:
 * 1. The MNIST digit reconstructions vs. the latent space
 * 2. The latent space values for the MNIST test set, as training progresses (every N minibatches)
 *
 * Note that for both plots, there is a slider at the top - change this to see how the reconstructions and latent
 * space changes over time.
 *
 * @author Alex Black
 */
public class VariationalAutoEncoderExample {

    static DataSetIterator _trainIter;
    static DataSetIterator _testIter;
    static MultiLayerNetwork _net;
    static org.deeplearning4j.nn.layers.variational.VariationalAutoencoder _vae;

    public static double minX = 1000000;
    public static double maxX = -1000000;
    public static double minY = 1000000;
    public static double maxY = -1000000;

    static int rngSeed = 12345;
    static int nEpochs = 3;

    static double plotMin = -40;                //Minimum values for plotting (x and y dimensions)
    static double plotMax = 40;                 //Maximum values for plotting (x and y dimensions)

    static INDArray testFeatures;
    static INDArray testLabels;
    static INDArray latentSpaceGrid;
    static List<INDArray> latentSpaceVsEpoch;
    static List<INDArray> digitsGrid;


    public static void main(String[] args) throws IOException {
        int minibatchSize = 10;

        //Plotting configuration
        int plotEveryNMinibatches = 10;    //Frequency with which to collect data for later plotting

        _trainIter = new AnomalyDataSetIterator(new ClassPathResource("OneClass/training_raul.csv").getFile().getPath(), minibatchSize);
        _testIter = new AnomalyDataSetIterator(new ClassPathResource("OneClass/eval_raul.csv").getFile().getPath(), minibatchSize);

        initializeModel();

        setData();

        //Add a listener to the network that, every N=100 minibatches:
        // (a) collect the test set latent space values for later plotting
        // (b) collect the reconstructions at each point in the grid
        _net.setListeners(new PlottingListener(plotEveryNMinibatches, testFeatures, latentSpaceGrid, latentSpaceVsEpoch, digitsGrid));

        trainModel();

        //Plot MNIST test set - latent space vs. iteration (every 100 minibatches by default)
        PlotUtil.plotData(latentSpaceVsEpoch, testLabels, plotMin, plotMax, plotEveryNMinibatches);


    }

    private static void setData() {
        int plotNumSteps = 16;
        //Test data for plotting
        DataSet testdata = _testIter.next(3000); //esto puedeser un ciclo
        testFeatures = testdata.getFeatures();
        testLabels = testdata.getLabels();
        latentSpaceGrid = getLatentSpaceGrid(plotMin, plotMax, plotNumSteps);              //X/Y grid values, between plotMin and plotMax

        //Lists to store data for later plotting
        latentSpaceVsEpoch = new ArrayList<>(nEpochs + 1);
        INDArray latentSpaceValues = _vae.activate(testFeatures, false, LayerWorkspaceMgr.noWorkspaces());                     //Collect and record the latent space values before training starts
        latentSpaceVsEpoch.add(latentSpaceValues);
        digitsGrid = new ArrayList<>();
    }

    private static void trainModel() {
        _trainIter.reset();
        //Perform training
        for (int i = 0; i < nEpochs; i++) {
            System.out.println(("Epoch: " + i));
            _net.pretrain(_trainIter);    //Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training
        }

        calculateArea();
    }

    private static void initializeModel() {
        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .list()
                .layer(0, new VariationalAutoencoder.Builder()
                        .activation(Activation.LEAKYRELU)
                        .encoderLayerSizes(5, 5)        //2 encoder layers, each of size 256
                        .decoderLayerSizes(5, 5)        //2 decoder layers, each of size 256
                        .pzxActivationFunction(Activation.IDENTITY)  //p(z|data) activation function
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
                        .nIn(10)                       //Input size: 28x28
                        .nOut(2)                            //Size of the latent variable space: p(z|x). 2 dimensions here for plotting, use more in general
                        .build())
                .pretrain(true).backprop(false).build();

        _net = new MultiLayerNetwork(conf);
        _net.init();

        //Get the variational autoencoder layer
        _vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) _net.getLayer(0);
    }

    private static void calculateArea() {
        _trainIter.reset();
        //_net.output(_trainIter, false); //Output = activate?
        INDArray latentSpaceValues = _vae.activate(_trainIter.next(1000).getFeatures(), false, LayerWorkspaceMgr.noWorkspaces());
        for (int i = 0; i < latentSpaceValues.length()/2; i++) {
            double x = latentSpaceValues.getDouble(i,0);
            double y = latentSpaceValues.getDouble(i,1);
            if (x < minX) {
                minX = x;
            }
            if (y < minY) {
                minY = y;
            }
            if (x > maxX) {
                maxX = x;
            }
            if (y > maxY) {
                maxY = y;
            }
        }
    }

    //This simply returns a 2d grid: (x,y) for x=plotMin to plotMax, and y=plotMin to plotMax
    private static INDArray getLatentSpaceGrid(double plotMin, double plotMax, int plotSteps) {
        INDArray data = Nd4j.create(plotSteps * plotSteps, 2);
        INDArray linspaceRow = Nd4j.linspace(plotMin, plotMax, plotSteps);
        for (int i = 0; i < plotSteps; i++) {
            data.get(NDArrayIndex.interval(i * plotSteps, (i + 1) * plotSteps), NDArrayIndex.point(0)).assign(linspaceRow);
            int yStart = plotSteps - i - 1;
            data.get(NDArrayIndex.interval(yStart * plotSteps, (yStart + 1) * plotSteps), NDArrayIndex.point(1)).assign(linspaceRow.getDouble(i));
        }
        return data;
    }

    private static class PlottingListener extends IterationListener {

        private final int plotEveryNMinibatches;
        private final INDArray testFeatures;
        private final INDArray latentSpaceGrid;
        private final List<INDArray> latentSpaceVsEpoch;
        private final List<INDArray> digitsGrid;
        private PlottingListener(int plotEveryNMinibatches, INDArray testFeatures, INDArray latentSpaceGrid,
                                 List<INDArray> latentSpaceVsEpoch, List<INDArray> digitsGrid){
            this.plotEveryNMinibatches = plotEveryNMinibatches;
            this.testFeatures = testFeatures;
            this.latentSpaceGrid = latentSpaceGrid;
            this.latentSpaceVsEpoch = latentSpaceVsEpoch;
            this.digitsGrid = digitsGrid;
        }

        @Override
        public void iterationDone(Model model, int iterationCount, int epoch) {
            if(!(model instanceof org.deeplearning4j.nn.layers.variational.VariationalAutoencoder)){
                return;
            }

            org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                    = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder)model;

            //Every N=100 minibatches:
            // (a) collect the test set latent space values for later plotting
            // (b) collect the reconstructions at each point in the grid
            if (iterationCount % plotEveryNMinibatches == 0) {
                INDArray latentSpaceValues = vae.activate(testFeatures, false, LayerWorkspaceMgr.noWorkspaces());
                latentSpaceVsEpoch.add(latentSpaceValues);

                INDArray out = vae.generateAtMeanGivenZ(latentSpaceGrid);
                digitsGrid.add(out);
            }
        }
    }
}
