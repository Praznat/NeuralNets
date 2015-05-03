package deepnets.testing;

import java.awt.*;

import deepnets.*;

import modeler.ModelLearnerHeavy;
import reasoner.*;
import reasoner.DecisionProcess.LogLevel;

public class CollectorGame extends GridExploreGame {

	private static boolean WRAP = false;
	private static boolean SEE_WALLS = true;
	private static int SIZE = 4;
	private static final String SAVE_NAME = "Collector"+SIZE+(SEE_WALLS?"Walls":"NoWalls");
	
	private ModelLearnerHeavy modeler;
	private final boolean[][] food;
	private boolean isLearning;
	private final RewardFunction SEEK = new RewardFunction() {
		@Override
		public double getReward(double[] stateVars) {
			double sum = 0;
			int n = playerGrid[0].length;
			for (int i = 0; i < playerGrid.length * n; i++) {
				final int c = i % cols;
				final int r = i / cols;
				sum += stateVars[i] * (food[c][r] ? 1 : 0);
			}
			return sum;
		}
	};
	private final RewardFunction MOVE_SEEK = new RewardFunction() {
		@Override
		public double getReward(double[] stateVars) {
			double sum = 0;
			for (int i = 0; i < stateVars.length; i++) {
				final int c = i % cols;
				final int r = i / cols;
				sum += stateVars[i] * ((food[c][r] ? 1 : 0) + (playerGrid[c][r] > 0.5 ? -.1 : 0));
			}
			return sum;
		}
	};
	
	public CollectorGame(int size) {
		super(size, size);
		food = new boolean[cols][rows];
		gridPanel.setGame(this);
	}
	public static CollectorGame trainedGame(int size, int learnIterations, int sampleSizeMultiplier, int m, int repaintMs) {
		CollectorGame game = new CollectorGame(size);
		FFNeuralNetwork storedNet = Utils.loadNetworkFromFile(SAVE_NAME);
		game.modeler = trainedModeler(size*size*m, size*size, game, storedNet == null ? sampleSizeMultiplier : 1, repaintMs,
				game.actionChoices, actionTranslator, new int[] {size*size*3});
		if (storedNet == null) {
			game.learnFromMemory(learnIterations);
			Utils.saveNetworkToFile(SAVE_NAME, game.modeler.getModelVTA().getNeuralNetwork());
		} else {
			game.modeler.getModelVTA().setANN(storedNet);
		}
		return game;
	}
	
	public static void main(String[] args) {
		test1();
		System.out.println("All done");
	}
	
	private static void test1() {
		int numSteps = SIZE * SIZE;
		int numRuns = 24;
		int joints = 1;
		int learnSteps = 50;
		int sampleSizeMultiplier = 500;
		int nodeMult = 5;
		int learnIterations = 200/nodeMult;
		int lilLearnIterations = 20/nodeMult;
		int epochs = 15000;
		CollectorGame game = trainedGame(SIZE, learnIterations, sampleSizeMultiplier, nodeMult, 2);

		game.createNewFoodPatch();
		game.createNewWalls();
		double skewFactor = 0.1;
		double discRate = 0.2;
		DecisionProcess decisionProcess = new DecisionProcess(game.modeler, game.actionChoices, numSteps,
				numRuns, joints, skewFactor, discRate);
		decisionProcess.setLogging(LogLevel.MID);
		Planner planner = Planner.createMonteCarloPlanner(game.modeler, numSteps, numRuns, game.SEEK,
				false, 0.25, joints, null, GridExploreGame.actionTranslator);
		Point lastPos = null;
		for (int i = 0; i < epochs; i++) {
			long startMs = System.currentTimeMillis();
			double explore = lastPos != null && game.playerPos.distance(lastPos) == 0 ? 0.9 : 0.1;
			lastPos = game.playerPos.getLocation();
			double[] preState = game.getState();
//			double[] actionNN = decisionProcess.buildDecisionTree(preState, game.SEEK, 3, 0, true, explore);
			double[] actionNN = planner.getOptimalAction(preState, game.actionChoices, explore, 0.5);
			game.modeler.observePreState(preState);
			game.modeler.observeAction(actionNN);
			game.move(actionTranslator.fromNN(actionNN), true);
			game.modeler.observePostState(game.getState());
			game.modeler.saveMemory(); //game.modeler.learnOnline(1.5, 0.5, 0);
			if ((i+1)%learnSteps == 0) game.learnFromMemory(lilLearnIterations);
			try {
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, 50 - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	protected void learnFromMemory(int learnIterations) {
		isLearning = true;
		modeler.learnFromMemory(1.5, 0.5, 0, false, learnIterations, 10000);
		System.out.println("mastery:	" + modeler.getPctMastered());
		modeler.clearExperience(); // ??
		isLearning = false;
	}
	
	boolean badfvmo = false;
	public void setupForTurn() { // dont rerandomize walls between pre-state and post-state
		if (badfvmo) createNewWalls();
		badfvmo = !badfvmo;
	}
	
	@Override
	public double[] getState() {
		double[] result = new double[rows * cols * (SEE_WALLS ? 2 : 1)];
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) result[c + r*cols] = playerGrid[c][r];
		}
		if (SEE_WALLS) {
			int k = rows * cols;
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) result[k + c + r*cols] = walls[c][r];
			}
		}
		return result;
	}
	
	@Override
	protected void move(double[] action, boolean repaint) {
		if (food[playerPos.x][playerPos.y]) {
			food[playerPos.x][playerPos.y] = false; // remove food after leaving food patch
			try { Thread.sleep(100); } catch (InterruptedException e) {}
			createNewFoodPatch();
			createNewWalls();
		}
		int origX = playerPos.x;
		int origY = playerPos.y;
		setPos(playerPos.x + (int) action[0], playerPos.y + (int) action[1], playerPos,playerGrid,WRAP,false);
		if (walls[playerPos.x][playerPos.y] > 0.5) setPos(origX, origY, playerPos,playerGrid,WRAP,false);
		if (repaint) frame.repaint();
	}

	protected void createNewWalls() {
		for (int c = 0; c < walls.length; c++) {
			int[] col = walls[c];
			for (int r = 0; r < col.length; r++) col[r] = 0;
		}
		Point wallPlace = (Point) playerPos.clone();
		for (int i = 0; i < Math.min(cols, rows)-2; i++) {
			createNewWall(wallPlace, i > 0, 10);
		}
	}
	protected void createNewWall(Point pos, boolean noEdges, int triesLeft) {
		if (triesLeft <= 0) return;
		int origX = pos.x;
		int origY = pos.y;
		int newX = pos.x + (int) (Math.random() * 3) - 1;
		int newY = pos.y + (int) (Math.random() * 3) - 1;
		setPos(newX, newY, pos,walls,WRAP,true);
		if (food[pos.x][pos.y] || playerGrid[pos.x][pos.y] > 0
				|| (noEdges && (pos.x == 0 || pos.x == cols-1 || pos.y == 0 || pos.y == rows-1))) {
			walls[pos.x][pos.y] = 0;
			pos.setLocation(origX, origY);
			createNewWall(pos, noEdges, triesLeft-1);
		}
	}
	
	protected void createNewFoodPatch() {
		int c = (int) (Math.random() * cols);
		int r = (int) (Math.random() * rows);
		if (food[c][r] || playerGrid[c][r] > 0.5) createNewFoodPatch(); // try again
		else {
			food[c][r] = true;
		}
	}
	
	@Override
	protected boolean showSmell() {
		return false;
	}
	
	@Override
	protected void paintGrid(Graphics g) {
		super.paintGrid(g);
		for (int c = 0; c < food.length; c++) {
			boolean[] col = food[c];
			for (int r = 0; r < col.length; r++) {
				if (walls[c][r] > 0) {
					g.setColor(Color.BLACK);
					g.fillRect(c*gUnit, r*gUnit, gUnit, gUnit);
					continue;
				}
				if (!food[c][r]) continue;
				g.setColor(new Color(0.0f,0.9f,0.0f,0.3f));
				g.fillRect(c*gUnit, r*gUnit, gUnit, gUnit);
			}
		}
		if (isLearning) {
			g.setColor(new Color(0.5f,0.5f,0.7f,0.3f));
			g.fillRect(0, 0, gUnit*cols, gUnit*rows);
		}
	}
}
