package transfertests;

import ann.FFNeuralNetwork;
import ann.indirectencodings.RelationManager;
import ann.testing.FlierCatcher;
import ann.testing.GridExploreGame;
import modeler.ModelLearnerHeavy;
import modularization.ModularizationUtils;

// results so far
// geonet is slow to train


public class T5GeoNet {

	final static int size = 5;
	final static int trainTurns = 500;
	final static int learnIterations = 200;
	
	public static void main(String[] args) {
		compareNormToGeo();
	}
	
	private static void compareNormToGeo() {
		FlierCatcher gameGeo = new FlierCatcher(size);
		makeModelerGeoNet(gameGeo, trainTurns, new int[] {5});
		System.out.println("Learning Geo Net");
		gameGeo.modeler.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);

		FlierCatcher gameNorm = new FlierCatcher(size);
		gameNorm.modeler = TransferTestUtils.loadOrCreate(null, gameGeo, trainTurns, new int[] {size*size*3}, null);
		System.out.println("Learning Normal Net");
		gameNorm.modeler.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
		
		TransferTestUtils.compareTwoModelers(gameNorm, 100, gameNorm.modeler, gameGeo.modeler, false);	
	}
	
	public static ModelLearnerHeavy makeModelerGeoNet(FlierCatcher game, int turns, int[] hiddenPerOutput) {
		game.modeler = TransferTestUtils.loadOrCreate(null, game, trainTurns, new int[] {size*size*3}, null);
		game.modeler.learnFromMemory(0, 0, 0, false, 1, 10000);
		RelationManager relMngr = RelationManager.createFromGridGamePredictor(game, game.modeler);
		FlierCatcher.trainModeler(game.modeler, turns, game, 0, game.actionChoices, GridExploreGame.actionTranslator);
		FFNeuralNetwork ann = game.modeler.getModelVTA().getNeuralNetwork();
		ModularizationUtils.initializeANNOnlyConnectingRelatedVars(ann, relMngr, hiddenPerOutput);
		return game.modeler;
	}
}
