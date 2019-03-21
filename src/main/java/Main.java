import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException, InterruptedException {

        /*Neural neural = new Neural();
        neural.train();
        neural.evaluate();*/

        Autoencoder autoencoder = new Autoencoder();
        autoencoder.train();
        autoencoder.evaluate();
    }
}
