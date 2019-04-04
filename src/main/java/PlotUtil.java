import org.nd4j.linalg.primitives.Pair;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Plotting methods for the VariationalAutoEncoder example
 * @author Alex Black
 */
public class PlotUtil {

    public static void plotData(List<INDArray> xyVsIter, INDArray labels, double axisMin, double axisMax, int plotFrequency){

        JPanel panel = new ChartPanel(createChart(xyVsIter.get(0), labels, axisMin, axisMax));
        JSlider slider = new JSlider(0,xyVsIter.size()-1,0);
        slider.setSnapToTicks(true);

        final JFrame f = new JFrame();
        slider.addChangeListener(new ChangeListener() {

            private JPanel lastPanel = panel;
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider slider = (JSlider)e.getSource();
                int  value = slider.getValue();
                JPanel panel = new ChartPanel(createChart(xyVsIter.get(value), labels, axisMin, axisMax));
                if(lastPanel != null){
                    f.remove(lastPanel);
                }
                lastPanel = panel;
                f.add(panel, BorderLayout.CENTER);
                f.setTitle("Salida bidimensional de la red en la interacion: " + value * plotFrequency);
                f.revalidate();
            }
        });

        f.setLayout(new BorderLayout());
        f.add(slider, BorderLayout.NORTH);
        f.add(panel, BorderLayout.CENTER);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Salida bidimensional de la red en la interacion: 0");

        f.setVisible(true);
    }



    //Test data
    private static XYDataset createDataSet(INDArray features, INDArray labelsOneHot){
        int nRows = features.rows();

        int nClasses = labelsOneHot.columns();

        XYSeries[] series = new XYSeries[nClasses];
        for( int i=0; i<nClasses; i++){
            series[i] = new XYSeries(String.valueOf(i));
        }
        INDArray classIdx = Nd4j.argMax(labelsOneHot, 1);
        for( int i=0; i<nRows; i++ ){
            int idx = classIdx.getInt(belongsTo(features.getDouble(i, 0), features.getDouble(i, 1)));
            series[idx].add(features.getDouble(i, 0), features.getDouble(i, 1));
        }

        XYSeriesCollection c = new XYSeriesCollection();
        for( XYSeries s : series) c.addSeries(s);
        return c;
    }

    private static int belongsTo(double x, double y) {
        if ((VariationalAutoEncoderExample.minX < x && VariationalAutoEncoderExample.maxX > x) &&
                (VariationalAutoEncoderExample.minY < y && VariationalAutoEncoderExample.maxY > y)) {
            return 0;
        }
        return 1;
    }

    private static JFreeChart createChart(INDArray features, INDArray labels, double axisMin, double axisMax) {
        return createChart(features, labels, axisMin, axisMax, "Vae: Salida 2D");
    }

    private static JFreeChart createChart(INDArray features, INDArray labels, double axisMin, double axisMax, String title ) {

        XYDataset dataset = createDataSet(features, labels);

        JFreeChart chart = ChartFactory.createScatterPlot(title,
                "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false);

        XYPlot plot = (XYPlot) chart.getPlot();
        plot.getRenderer().setBaseOutlineStroke(new BasicStroke(0));
        plot.setNoDataMessage("NO DATA");

        plot.setDomainPannable(false);
        plot.setRangePannable(false);
        plot.setDomainZeroBaselineVisible(true);
        plot.setRangeZeroBaselineVisible(true);

        plot.setDomainGridlineStroke(new BasicStroke(0.0f));
        plot.setDomainMinorGridlineStroke(new BasicStroke(0.0f));
        plot.setDomainGridlinePaint(Color.blue);
        plot.setRangeGridlineStroke(new BasicStroke(0.0f));
        plot.setRangeMinorGridlineStroke(new BasicStroke(0.0f));
        plot.setRangeGridlinePaint(Color.blue);

        plot.setDomainMinorGridlinesVisible(true);
        plot.setRangeMinorGridlinesVisible(true);

        XYLineAndShapeRenderer renderer
                = (XYLineAndShapeRenderer) plot.getRenderer();
        renderer.setSeriesOutlinePaint(0, Color.black);
        renderer.setUseOutlinePaint(true);
        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setAutoRangeIncludesZero(false);
        domainAxis.setRange(axisMin, axisMax);

        domainAxis.setTickMarkInsideLength(2.0f);
        domainAxis.setTickMarkOutsideLength(2.0f);

        domainAxis.setMinorTickCount(2);
        domainAxis.setMinorTickMarksVisible(true);

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setTickMarkInsideLength(2.0f);
        rangeAxis.setTickMarkOutsideLength(2.0f);
        rangeAxis.setMinorTickCount(2);
        rangeAxis.setMinorTickMarksVisible(true);
        rangeAxis.setRange(axisMin, axisMax);
        return chart;
    }
}
