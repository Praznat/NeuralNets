package modulemanagement;

import java.awt.Color;
import java.awt.Graphics;
import java.util.HashMap;
import java.util.Map;

import ann.FFNeuralNetwork;
import ann.Utils;
import ann.testing.IGridGame;
import modeler.ModelLearnerModularPure;
import modularization.WeightPruner;

public class ModuleDisplayer extends ModelDisplayer<ModelLearnerModularPure> {

	private static final Color[] COLORS = new Color[] {
			Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.CYAN, Color.MAGENTA, Color.ORANGE, Color.PINK,
			new Color(128, 0, 128)
	};
	private Map<Integer, ModuleDistribution<Integer>> nodeModules;
	private Map<ReusableIntModule, Color> modulesColored = new HashMap<ReusableIntModule, Color>();

	public static ModuleDisplayer create(ModelLearnerModularPure modeler,
			final int numStateVars, final IGridGame game) {
		int area = game.getRows() * game.getCols();
		int numObjs = numStateVars / area;
		int[][] grids = new int[numObjs + 1][];
		for(int i = 0; i < grids.length-1; i++) grids[i] = new int[] {game.getRows(), game.getCols()};
		grids[numObjs] = new int[] {numStateVars % area, 1};
		return new ModuleDisplayer(modeler, numObjs + 1, grids);
	}
	
	public ModuleDisplayer(ModelLearnerModularPure modeler, int actGrid, int[][] grids) {
		super(modeler, actGrid, grids);
		setup();
	}
	
	private void setup() {
		nodeModules = modeler.getModuleManager().getNodeModules();
	}
	
	// paint by module used
	@Override
	protected void paintPredVar(int r, int c, Graphics g, int v) {
		super.paintPredVar(r, c, g, v);
		ModuleDistribution<Integer> dist = nodeModules.get(v);
		ReusableIntModule module = getModuleFromDist(dist);
		g.setColor(getColor(module));
		g.fillRect(c*gUnit+1, r*gUnit+1, gUnit-2, gUnit-2);
	}
	
	private static ReusableIntModule getModuleFromDist(ModuleDistribution<Integer> dist) {
		return (ReusableIntModule) (dist != null ? dist.getMostLikelyModule() : null);
//		return (ReusableIntModule) (dist != null ? dist.drawModuleProbabilistically() : null);
	}
	
	private Color getColor(ReusableIntModule module) {
		if (module == null) return Color.WHITE;
		Color c = modulesColored.get(module);
		if (c == null) { // module not seen yet
			int i = modulesColored.size();
			if (i >= COLORS.length) c = new Color((int) (256 * Math.random()),
					(int) (256 * Math.random()), (int) (256 * Math.random()));
			else c = COLORS[i];
			modulesColored.put(module, c);
		}
		return c;
	}
	
	@Override
	protected void reportOnOutputVar(final int varKey) {
		// TODO draw module neural net? print wgt matrices?
		System.out.println(varKey + "th output");
		ModuleDistribution<Integer> dist = nodeModules.get(varKey);
		dist.report(10);
	}

	@Override
	protected ModulePainter createModulePainter(final int varKey) {
		return new ModulePainter() {
			private double[][] inOutAbsConnWgt;
			private double maxWeight;
			@Override
			public void modulePaint(Graphics g) {
				FFNeuralNetwork nn = modeler.getTransitionsModule().getNeuralNetwork();
				if (inOutAbsConnWgt == null) {
					inOutAbsConnWgt = WeightPruner.inOutAbsConnWgt(nn, false);
					maxWeight = Utils.max(inOutAbsConnWgt);
				}
				final double[] varPower = new double[inOutAbsConnWgt.length];
				for (int i = 0; i < varPower.length; i++) {
					double w = inOutAbsConnWgt[i][varKey];
					varPower[i] = w;
				}
				final double denom = maxWeight;
				GridPainter gp = new GridPainter() {
					int j = 0;
					@Override
					public void paintCell(Graphics g, int c, int r) {
						g.setColor(Color.BLACK);
						g.drawRect(c*gUnit, r*gUnit, gUnit, gUnit);
						float w = (float) (varPower[j++] / denom);
						g.setColor(new Color(1f, 0f, 0f, w));
						g.fillRect(c*gUnit, r*gUnit, gUnit, gUnit);
					}
				};
				paintGridCells(g, gp);
			}
		};
	}
}
