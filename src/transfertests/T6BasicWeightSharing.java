package transfertests;

import java.util.Collection;

import ann.ActivationFunction;
import ann.Connection;
import ann.testing.FlierCatcher;
import ann.testing.GridExploreGame;
import modeler.ModelLearner;
import modeler.ModelLearnerHeavy;
import modularization.HardWeightSharing;
import modularization.ModuleDisplayer;

public class T6BasicWeightSharing {

	final static int size = 5;
	final static int trainTurns = 500;
	final static int learnIterations = 200;

	public static void main(String[] args) {
		testDebug();
	}
	
	private static void testDebug() {
		FlierCatcher game = new FlierCatcher(size);
		ModelLearner model = createModelerWithWgtSharing("T6DEBUG", game);
		model.learnFromMemory(0.9, 0.5, 0, false, learnIterations, 10000);
		Collection<Connection> allConnections = Connection.getAllConnections(model.getTransitionsModule().getNeuralNetwork());
		for (Connection conn : allConnections) {
			System.out.println(conn.getWeight().getWeight());
		}
		int[] gridSpec = new int[] {game.rows, game.cols};
	
		int epochs = 100;
		int numSteps = 3;
		int numRuns = 5;
		int joints = 1;
		double skewFactor = 0.1;
		double discRate = 0.2;
		game.repaintMs = 70;
		FlierCatcher.play((ModelLearnerHeavy)model, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);		
		ModuleDisplayer moduleDisplayer = new ModuleDisplayer(model, 2, gridSpec, gridSpec, new int[] {4, 1});
	}

	static ModelLearnerHeavy createModelerWithWgtSharing(String saveName, FlierCatcher game) {
		return createModelerWithWgtSharing(saveName, game, trainTurns, false);
	}
	static ModelLearnerHeavy createModelerWithWgtSharing(String saveName, FlierCatcher game, int turns, boolean actuallyDontWgtShare) {
		HardWeightSharing hws = actuallyDontWgtShare ? null : new HardWeightSharing(new double[] {-.5, 0, .5}, 8, 0.9, 0.5, 0);
		game.modeler = new ModelLearnerHeavy(500, new int[] {size*size*2}, null, null, ActivationFunction.SIGMOID0p5, turns);
		FlierCatcher.trainModeler(game.modeler, turns, game, 0, game.actionChoices, GridExploreGame.actionTranslator);
		game.modeler.getModelVTA().setWgtSharer(hws);
		return game.modeler;
	}

}
