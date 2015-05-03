package deepnets.testing;

import java.awt.*;
import java.util.ArrayList;

import modeler.*;
import reasoner.*;
import transfer.ReuseNetwork;
import utils.RandomUtils;
import deepnets.*;


public class AxisTransfer extends GridExploreGame {

	private static final boolean HAVE_ENEMIES = true;
	private static final double SPAWN_FREQ = 0.35;
	private static boolean VERT_NOT_HORZ = true;
	
	private boolean endSwitch = true;
	private final RewardFunction SEEK_EDGE = new RewardFunction() {
		@Override
		public double getReward(double[] stateVars) {
			double sum = 0;
			int n = playerGrid[0].length;
			for (int i = 0; i < playerGrid.length * n; i++) {
				final int c = i % cols;
				final int r = i / cols;
				sum += stateVars[i] * (goalReached(c, r) ? 1 : 0);
			}
			return sum;
		}
	};
	private final RewardFunction SEEK_ENEMY = GridTagGame.tagReward(true, cols, rows);
	private static String SAVE_NAME = "";
	private final int[][] opponentGrid;
	private ModelLearnerHeavy modeler;
	private int eats = 0;
	private int misses = 0;

	private boolean goalReached(int c, int r) {
		int x = VERT_NOT_HORZ ? r : c;
		return x == (endSwitch ? 0 : (cols-1));
	}

	public AxisTransfer(int size) {
		super(size-1, size+1);
		opponentGrid = new int[cols][rows];
		gridPanel.setGame(this);
	}

	public static void main(String[] args) {
		testModularization(-.7);
		
		testScratch();
		epochs = 1;
		testSandwiches();
	}

	private static int size = 5;
	private static int numSteps = size-3;
	private static int numRuns = 10;
	private static int joints = 1;
	private static double skewFactor = 0.1;
	private static double discRate = 0.2;
	private static double confusion = 0.1;
	private static int epochs = 500;
	private static boolean useRollouts = true;

	private static void testModularization(double stdevsBelow) {
		int sampleSizeMultiplier = 50;
		int learnIterations = 5;
		epochs = 10;
		AxisTransfer game = playSourceDomain(size, sampleSizeMultiplier, learnIterations);
//		WeightPruner.inOutAbsConnWgt(game.modeler.getModelJDM().getNeuralNetwork());
		WeightPruner.pruneBelowAvg(game.modeler.getModelVTA().getNeuralNetwork(), stdevsBelow);
		WeightPruner.pruneBelowAvg(game.modeler.getModelJDM().getNeuralNetwork(), stdevsBelow);
		game.modeler.getModelJDM().toggleShouldDisconnect(false);
		game.modeler.learnFromMemory(1.5, 0.5, 0, false, learnIterations, 10000);
//		WeightPruner.inOutAbsConnWgt(game.modeler.getModelJDM().getNeuralNetwork());
		epochs = 500;
		play(game.modeler, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
		
	}
	private static void testScratch() {
		int sampleSizeMultiplier = 250;
		int learnIterations = 5;
		ArrayList<Double> results = playSourceDomain(size, sampleSizeMultiplier, learnIterations).modeler.getTrainingLog();
		for (double d: results) System.out.println(d);
	}
	public static void testSandwiches() {
		int sampleSizeMultiplier = 5;
		int learnIterations = 50;
		VERT_NOT_HORZ = !VERT_NOT_HORZ;
		ArrayList<Double> baddie = playUsingSandwichedTarget(SAVE_NAME, true, size, sampleSizeMultiplier, learnIterations);
		ArrayList<Double> goodie = playUsingSandwichedTarget(SAVE_NAME, false, size, sampleSizeMultiplier, learnIterations);
		int n = Math.min(baddie.size(), goodie.size());
		System.out.println("Random	Transfered");
		for (int i = 0; i < n; i++) System.out.println(baddie.get(i) + "	" + goodie.get(i));
	}
	private static ArrayList<Double> playUsingSandwichedTarget(String targetNet, boolean randomizeTarget,
			int size, int sampleSizeMultiplier, int learnIterations) {
		FFNeuralNetwork storedNet = Utils.loadNetworkFromFile(targetNet);
		if (storedNet == null) {
			System.out.println("No save data so no transfer");
			return null;
		}
		ReuseNetwork target = ReuseNetwork.createSandwichedNetwork(storedNet, true);
		if (randomizeTarget) RandomUtils.randomizeWeights(target);
		AxisTransfer game = new AxisTransfer(size);
		ModelLearnerHeavy modeler = trainedModeler(size*size, size, game, sampleSizeMultiplier, 0,
				game.actionChoices, actionTranslator, null);
		modeler.learnFromMemory(0.1, 0, 0, false, 1); // just to adjust size etc
		modeler.getModelVTA().setANN(target);
		int turns = size * sampleSizeMultiplier;
		System.out.println("Training " + (randomizeTarget ? "random" : "") + "sandwich");
		trainModeler(modeler, turns, game, 1, game.actionChoices, actionTranslator);
		modeler.recordTraining();
		modeler.learnFromMemory(1.5, 0.5, 0, false, learnIterations, 10000);
		play(modeler, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
		return modeler.getTrainingLog();
	}
	private static AxisTransfer playSourceDomain(int size, int sampleSizeMultiplier, int learnIterations) {
		AxisTransfer game = new AxisTransfer(size);
		game.modeler = trainedModeler(size*size*2, size*size*2, game, sampleSizeMultiplier, 0,
				game.actionChoices, actionTranslator, new int[] {size*size*3});
		boolean loaded = Utils.loadModelerFromFile(game.modeler, SAVE_NAME);
		if (!loaded) {
			game.modeler.recordTraining();
			game.modeler.learnFromMemory(1.5, 0.5, 0, false, learnIterations, 10000);
			Utils.saveModelerToFile(SAVE_NAME, game.modeler);
		}
		play(game.modeler, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
		return game;
	}
	private static void play(ModelLearnerHeavy modeler, AxisTransfer game, int epochs, int numSteps, int numRuns,
			int joints, double skewFactor, double discRate) {
		DecisionProcess decisionProcess = new DecisionProcess(modeler, game.actionChoices, numSteps,
				numRuns, joints, skewFactor, discRate);
		Planner planner = Planner.createMonteCarloPlanner(modeler, numSteps, numRuns, game.SEEK_ENEMY,
				false, discRate, joints, null, GridExploreGame.actionTranslator);
		game.eats = 0;
		game.misses = 0;
		for (int i = 0; i < epochs; i++) {
			long startMs = System.currentTimeMillis();
			double[] preState = game.getState();
			double[] actionNN = useRollouts ? planner.getOptimalAction(preState, game.actionChoices, 0.01, 0.5)
			: decisionProcess.buildDecisionTree(preState, game.getRewFn(), numSteps, 0, true, confusion);
			modeler.observePreState(preState);
			modeler.observeAction(actionNN);
			game.move(actionTranslator.fromNN(actionNN), true);
			System.out.println(i + "	" + game.eats + "	" + game.misses);
			modeler.observePostState(game.getState());
			modeler.saveMemory(); //game.modeler.learnOnline(1.5, 0.5, 0);
			try {
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, 50 - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	private RewardFunction getRewFn() {
		return HAVE_ENEMIES ? SEEK_ENEMY : SEEK_EDGE;
	}

	@Override
	protected void move(double[] action, boolean repaint) {
		setPos(playerPos.x + (VERT_NOT_HORZ ? 0 : (int) action[0]),
				playerPos.y + (VERT_NOT_HORZ ? (int) action[1] : 0),
				playerPos,playerGrid,false,false);
		if (goalReached(playerPos.x, playerPos.y)) {
//			System.out.println("Goal reached.");
			endSwitch = !endSwitch;
		} else {
//			System.out.println("---");
		}
		if (repaint) frame.repaint();
		if (HAVE_ENEMIES) moveEnemies();
	}

	protected void moveEnemies() {
		if (VERT_NOT_HORZ) {
			int x = (int) (Math.random() * rows);
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols-1; c++) {
					opponentGrid[c][r] = opponentGrid[c+1][r];
				}
				opponentGrid[cols-1][r] = 0;
				if (r == x && Math.random() < SPAWN_FREQ && opponentGrid[cols-2][r] < 0.5)
					opponentGrid[cols-1][r] = 1;
			}
		} else {
			int x = (int) (Math.random() * cols);
			for (int c = 0; c < cols; c++) {
				for (int r = 0; r < rows-1; r++) {
					opponentGrid[c][r] = opponentGrid[c][r+1];
				}
				opponentGrid[c][rows-1] = 0;
				if (c == x && Math.random() < SPAWN_FREQ && opponentGrid[c][rows-2] < 0.5)
					opponentGrid[c][rows-1] = 1;
			}
		}
	}

	@Override
	public double[] getState() {
		double[] result = new double[rows * cols * (HAVE_ENEMIES ? 2 : 1)];
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) result[c + r*cols] = playerGrid[c][r];
		}
		if (HAVE_ENEMIES) {
			int k = rows * cols;
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) result[k + c + r*cols] = opponentGrid[c][r];
			}
		}
		return result;
	}

	@Override
	protected boolean showSmell() {
		return false;
	}

	@Override
	protected void paintGrid(Graphics g) {
		super.paintGrid(g);
		final int gSub = gUnit/4;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (opponentGrid[c][r] > 0.5) {
					g.setColor(Color.BLACK);
					g.fillRect(c*gUnit+gSub*2/3, r*gUnit + (gUnit-gSub)/2, gSub*2, gSub);
//					g.drawLine(c*gUnit+gSub, r*gUnit+gSub, (c+1)*gUnit-gSub, (r+1)*gUnit-gSub);
//					g.drawLine(c*gUnit+gSub, (r+1)*gUnit-gSub, (c+1)*gUnit-gSub, r*gUnit+gSub);
					if (playerGrid[c][r] > 0.5) {
						g.setColor(Color.BLUE);
						g.fillOval(c*gUnit+gSub, r*gUnit+gSub, gUnit - 2*gSub, gUnit - 2*gSub);
						eats++;
					} else if (c == 0) {
						misses++;
					}
				}
			}
		}
	}
}
