package ann.testing;

import java.awt.Color;
import java.awt.Graphics;
import java.util.ArrayList;
import java.util.List;

import ann.ActivationFunction;
import ann.Utils;
import modeler.ModelLearner;
import modeler.ModelLearnerHeavy;
import modularization.WeightPruner;
import reasoner.DecisionProcess;
import reasoner.Planner;
import reasoner.RewardFunction;
import transfer.ReuseNetwork;
import utils.RandomUtils;


public class FlierCatcher extends GridExploreGame {

	public enum MoveRule {NORMAL, CELLAUTONAMA};
	
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
	private static final String SAVE_NAME = "FlyCells";
	private final int[][] opponentGrid;
	private boolean endSwitch = true;
	private boolean haveEnemies = true;
	private double spawnFreq = 0.35;
	private MoveRule moveRule = MoveRule.NORMAL;
	private boolean playerMoveVertNotHorz = true;
	private int eats = 0;
	private int misses = 0;
	private RewardFunction rewardFnContainer = new RewardFunction() {
		@Override
		public double getReward(double[] stateVars) {
			return rewardFn.getReward(stateVars);
		}
	};

	private boolean goalReached(int c, int r) {
		int x = playerMoveVertNotHorz ? r : c;
		return x == (endSwitch ? 0 : (cols-1));
	}

	public FlierCatcher(int size) {
		super(size-1, size+1);
		opponentGrid = new int[cols][rows];
		gridPanel.setGame(this);
		controlPanel.setGame(this);
		rewardFn = GridTagGame.follow(this);
	}

	public static void main(String[] args) {
//		testOnline();
		
		testScratch();
		
//		testModularization(-.7);
		
//		for (int i = 0; i < 50; i++) testSandwiches();
//		for (String s : loggy) System.out.println(s);
		
//		game = new FlierCatcher(size);
//		int turns = 5;
//		game.modeler = new ModelLearnerHeavy(500, new int[] {size*size*2},
//				null, new int[] {size*size*3}, ActivationFunction.SIGMOID0p5, turns);
//		trainModeler(game.modeler, turns, game, repaintMs, game.actionChoices, actionTranslator);
//		ModelSelector sourcer = new ModelSelector("Flies", "FlyCells");
//		SimulationScore ss = sourcer.observeWorkingModelTransitions(game.modeler, turns - 1);
//		System.out.println(ss);
	}

	private static int size = 5;
	private static int numSteps = size-3;
	private static int numRuns = 10;
	private static int joints = 1;
	private static double skewFactor = 0.1;
	private static double discRate = 0.2;
	private static double confusion = 0.1;
	private static int epochs = 50;
	private static boolean useRollouts = true;
	public static int repaintMs = 1;
	private static FlierCatcher game;
	private static List<String> loggy = new ArrayList<String>();
	
	private static void play() {
		play(game.modeler, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
	}
	
	private static void testOnline() {
		epochs = 1000;
		int learnIterations = 100;
		FlierCatcher game = new FlierCatcher(size);
		ArrayList<double[]> acts = new ArrayList<double[]>();
		acts.add(game.actionChoices.get(2));
		acts.add(game.actionChoices.get(3));
		game.modeler = new ModelLearnerHeavy(500, new int[] {}, null, null, ActivationFunction.SIGMOID0p5, epochs);
//		game.modeler.getModelJDM().toggleShouldDisconnect(false);
		int lastTrain = 0;
		for (int i = 0; i < epochs; i++) {
			trainModeler(game.modeler, 1, game, learnIterations, acts, actionTranslator);
			if (i > lastTrain * 1.5) {
				game.modeler.learnFromMemory(1.9, 0.5, 0, false, learnIterations, 10000);
				lastTrain = i;
			}
		}
	}
	
	private static void testModularization(double stdevsBelow) {
		int sampleSizeMultiplier = 50;
		int learnIterations = 5;
		epochs = 10;
		FlierCatcher game = playSourceDomain(size, sampleSizeMultiplier, learnIterations);
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
		int turns = 2000;
		int learnIterations = 500;
		FlierCatcher game = new FlierCatcher(size);
//		game.actionChoices.remove(1);
//		game.actionChoices.remove(0);
		game.modeler = new ModelLearnerHeavy(500, new int[] {size*size*2}, null, null, ActivationFunction.SIGMOID0p5, turns);
		trainModeler(game.modeler, turns, game, repaintMs, game.actionChoices, actionTranslator);
		game.modeler.learnFromMemory(0.3, 0.5, 0, false, learnIterations, 10000);
		repaintMs = 70;
		play(game.modeler, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
	}
	public static void testSandwiches() {
		double sampleSizeMultiplier = 3;
		int learnIterations = 50;
//		VERT_NOT_HORZ = !VERT_NOT_HORZ;
		epochs = 100;
		ArrayList<Double> goodie = playUsingSandwichedTarget(SAVE_NAME, false, size, sampleSizeMultiplier, learnIterations);
		double g = game.getWinRate();
		ArrayList<Double> baddie = playUsingSandwichedTarget(SAVE_NAME, true, size, sampleSizeMultiplier, learnIterations);
		double b = game.getWinRate();
		loggy.add(g + " vs	" + b);
//		int n = Math.min(baddie.size(), goodie.size());
//		System.out.println("Random	Transfered");
//		for (int i = 0; i < n; i++) System.out.println(baddie.get(i) + "	" + goodie.get(i));
	}
	
	public double getWinRate() {
		return (eats + 0.0) / (eats + misses);
	}
	
	public void clearWinRate() {
		eats = 0;
		misses = 0;
	}
	
	private static ArrayList<Double> playUsingSandwichedTarget(String targetNet, boolean randomizeTarget,
			int size, double sampleSizeMultiplier, int learnIterations) {
		int turns = (int) (size * sampleSizeMultiplier);
		ModelLearner storedModeler = Utils.loadModelerFromFile(targetNet, turns);
		if (storedModeler == null) {
			System.out.println("No save data so no transfer");
			return null;
		}
		ReuseNetwork targetP = ReuseNetwork.createSandwichedNetwork(storedModeler.getTransitionsModule().getNeuralNetwork(), true);
		ReuseNetwork targetC = ReuseNetwork.createSandwichedNetwork(storedModeler.getFamiliarityModule().getNeuralNetwork(), true);
		if (randomizeTarget) {
			RandomUtils.randomizeWeights(targetP);
			RandomUtils.randomizeWeights(targetC);
		}
		game = new FlierCatcher(size);
		ModelLearnerHeavy modeler = trainedModeler(size*size, size, game, sampleSizeMultiplier, 0,
				game.actionChoices, actionTranslator, new int[] {size*size*3});
		modeler.learnFromMemory(0.1, 0, 0, false, 1); // just to adjust size etc
		modeler.getModelVTA().setANN(targetP);
		modeler.getModelJDM().setANN(targetC);
		System.out.println("Training " + (randomizeTarget ? "random" : "") + "sandwich");
		trainModeler(modeler, turns, game, 1, game.actionChoices, actionTranslator);
		modeler.recordTraining();
		modeler.learnFromMemory(1.5, 0.5, 0, false, learnIterations, 10000);
		repaintMs = 1;
		play(modeler, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
		return modeler.getTrainingLog();
	}
	private static FlierCatcher playSourceDomain(int size, int sampleSizeMultiplier, int learnIterations) {
		FlierCatcher game = new FlierCatcher(size);
		game.modeler = trainedModeler(size*size*2, size*size*2, game, sampleSizeMultiplier, 0,
				game.actionChoices, actionTranslator, new int[] {size*size*6});
		boolean loaded = Utils.loadModelerFromFile(game.modeler, SAVE_NAME);
		if (!loaded) {
			game.modeler.recordTraining();
			game.modeler.learnFromMemory(1.5, 0.5, 0, false, learnIterations, 10000);
			Utils.saveModelerToFile(SAVE_NAME, game.modeler);
		}
		play(game.modeler, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
		return game;
	}
	public static void play(ModelLearnerHeavy modeler, FlierCatcher game, int epochs, int numSteps, int numRuns,
			int joints, double skewFactor, double discRate) {
		DecisionProcess decisionProcess = new DecisionProcess(modeler, game.actionChoices, numSteps,
				numRuns, joints, skewFactor, discRate);
		Planner planner = Planner.createMonteCarloPlanner(modeler, numSteps, numRuns, game.rewardFnContainer ,
				false, discRate, joints, null, GridExploreGame.actionTranslator);
		game.eats = 0;
		game.misses = 0;
		for (int i = 0; i < epochs; i++) {
			long startMs = System.currentTimeMillis();
			double[] preState = game.getState();
			double[] actionNN = useRollouts ? planner.getOptimalAction(preState, game.actionChoices, 0.01, 0.5)
			: decisionProcess.buildDecisionTree(preState, game.rewardFnContainer, numSteps, 0, true, confusion);
			modeler.observePreState(preState);
			modeler.observeAction(actionNN);
			game.move(actionTranslator.fromNN(actionNN), true);
			if (repaintMs > 0) System.out.println(i + "	" + game.eats + "	" + game.misses);
			modeler.observePostState(game.getState());
			modeler.saveMemory(); //game.modeler.learnOnline(1.5, 0.5, 0);
			try {
				synchronized (game.thread) {
					while (game.isPaused) {
						game.thread.wait();
					}
				}
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, repaintMs - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	private RewardFunction getRewFn() {
		return haveEnemies ? rewardFn : SEEK_EDGE;
	}

	@Override
	protected void move(double[] action, boolean repaint) {
		setPos(playerPos.x + (playerMoveVertNotHorz ? 0 : (int) action[0]),
				playerPos.y + (playerMoveVertNotHorz ? (int) action[1] : 0),
				playerPos,playerGrid,false,false);
		if (goalReached(playerPos.x, playerPos.y)) {
//			System.out.println("Goal reached.");
			endSwitch = !endSwitch;
		} else {
//			System.out.println("---");
		}
		if (repaint) frame.repaint();
		if (haveEnemies) moveEnemies();
	}
	
	private int enemyMoveRule(int[] before, int x) {
		switch(getMoveRule()) {
		default:
		case NORMAL: return before[x];
		case CELLAUTONAMA:
			int n = rotateArray(before, x, -1) + rotateArray(before, x, 1) + before[x];
			return n == 2 ? 1 : 0;
		}
	}
	
	private int rotateArray(int[] a, int curr, int offset) {
		return a[(curr + offset + a.length) % a.length];
	}

	protected void moveEnemies() {
		if (playerMoveVertNotHorz) {
			int x = (int) (Math.random() * rows);
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols-1; c++) {
					opponentGrid[c][r] = enemyMoveRule(opponentGrid[c+1], r);
				}
				opponentGrid[cols-1][r] = 0;
				if ((r == x || getMoveRule() == MoveRule.CELLAUTONAMA)
						&& Math.random() < spawnFreq && opponentGrid[cols-2][r] < 0.5)
					opponentGrid[cols-1][r] = 1;
				if (opponentGrid[0][r] > 0.5) {
					if (playerGrid[0][r] > 0.5) eats++;
					else misses++;
				}
			}
		} else {
			int x = (int) (Math.random() * cols);
			for (int c = 0; c < cols; c++) {
				for (int r = 0; r < rows-1; r++) {
					opponentGrid[c][r] = enemyMoveRule(opponentGrid[c], r+1);
				}
				opponentGrid[c][rows-1] = 0;
				if ((c == x || getMoveRule() == MoveRule.CELLAUTONAMA)
						&& Math.random() < spawnFreq && opponentGrid[c][rows-2] < 0.5)
					opponentGrid[c][rows-1] = 1;
				if (opponentGrid[c][0] > 0.5) {
					if (playerGrid[c][0] > 0.5) eats++;
					else misses++;
				}
			}
		}
	}

	@Override
	public double[] getState() {
		double[] result = new double[rows * cols * (haveEnemies ? 2 : 1)];
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) result[c + r*cols] = playerGrid[c][r];
		}
		if (haveEnemies) {
			int k = rows * cols;
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) result[k + c + r*cols] = opponentGrid[c][r];
			}
		}
		return result;
	}
	
	public double[] getBlurredState() {
		// TODO
		return getState();
	}

	@Override
	protected boolean showSmell() {
		return false;
	}
	
	@Override
	public void convertFromState(double[] state) {
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) playerGrid[c][r] = (int) Math.round(state[c + r*cols]);
		}
		if (!haveEnemies) return;
		final int k = rows * cols;
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) opponentGrid[c][r] = (int)Math.round(state[k + c + r*cols]);
		}
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
					}
				}
			}
		}
	}

	public MoveRule getMoveRule() {
		return moveRule;
	}

	public void setMoveRule(MoveRule moveRule) {
		this.moveRule = moveRule;
	}

	public double getSpawnFreq() {
		return spawnFreq;
	}

	public void setSpawnFreq(double spawnFreq) {
		this.spawnFreq = spawnFreq;
	}
}
