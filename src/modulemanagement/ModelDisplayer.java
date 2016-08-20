package modulemanagement;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.HashMap;
import java.util.Map;

import javax.swing.JFrame;
import javax.swing.JPanel;

import ann.FFNeuralNetwork;
import ann.Utils;
import modeler.ModelLearner;
import modeler.ModelLearnerModularImpure;
import modeler.ModelLearnerModularPure;
import modularization.WeightPruner;

public class ModelDisplayer<T extends ModelLearner> implements MouseListener {

	private JFrame frame;
	private ModulePanel inPanel;
	private ModulePanel outPanel;
	protected T modeler;
	protected int[][] grids;
	protected int numVars;
	protected int gUnit;
	private Map<Point, Integer> centerToVar = new HashMap<Point, Integer>();

	public ModelDisplayer(T modeler, final int actGridKey, final int[]... grids) {
		this.grids = grids;
		this.numVars = countVars();
		gUnit = (int) Math.round(400 / Math.sqrt(numVars));
		this.modeler = modeler;
		if (this.modeler instanceof ModelLearnerModularImpure) {
			System.out.println("WARNING: Consider useing ModuleDisplayer for ModelLearnerModular");
		}
		createJFrame();
		final GridPainter gp = new GridPainter() {
			private int v = 0;
			@Override
			public void paintCell(Graphics g, int c, int r) {
				super.paintCell(g, c, r);
				if (v < numVars) {
					paintPredVar(r, c, g, v++);
				} else {
					System.out.println("ModelDisplayer shouldnt count v farther than numVars");
				}
			}
		};
		this.inPanel.setModulePainter(new ModulePainter() {
			@Override
			public void modulePaint(Graphics g) {
				paintGridCells(g, gp, actGridKey);
			}
		});
	}
	
	protected void paintPredVar(int r, int c, Graphics g, int v) {
		Point p = new Point(c*gUnit + gUnit/2, r*gUnit + gUnit/2);
		centerToVar.put(p, v);
	}
	
	protected void reportOnOutputVar(final int varKey) {
		outPanel.setModulePainter(createModulePainter(varKey));
		outPanel.repaint();
	}

	protected void paintGridCells(Graphics g, GridPainter gp) {
		paintGridCells(g, gp, -1);
	}
	// skipGrid indicates index of skip action vars grid in predicted vars
	protected void paintGridCells(Graphics g, GridPainter gp, int skipGrid) {
		int offC = 0;
		int i = 0;
		for (int[] grid : grids) {
			if (i++ == skipGrid) continue;
			int rows = grid[0];
			int cols = grid[1];
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					gp.paintCell(g, c + offC, r);
				}
			}
			offC += cols + 1;
		}
	}
	
	private void createJFrame() {
		boolean hasSensitivityGraph = !modeler.getClass().isAssignableFrom(ModelLearnerModularPure.class);
		frame = new JFrame();
		inPanel = new ModulePanel();
		outPanel = new ModulePanel();
		frame.setLayout(new GridLayout(0,hasSensitivityGraph ? 2 : 1));
		frame.add(inPanel);
		inPanel.addMouseListener(this);
		if (hasSensitivityGraph) frame.add(outPanel);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.pack();
	}
	
	private int countVars() {
		int n = 0;
		for (int[] grid : grids) {
			n += grid[0] * grid[1];
		}
		return n;
	}
	
	@Override
	public void mouseClicked(MouseEvent arg0) {
		// TODO Auto-generated method stub
		Point m = new Point(arg0.getX(), arg0.getY());
		double closestD = 666666;
		Point closestP = null;
		for (Point p : centerToVar.keySet()) {
			double d = m.distanceSq(p);
			if (d < closestD) {
				closestD = d;
				closestP = p;
			}
		}
		int varKey = centerToVar.get(closestP);
		reportOnOutputVar(varKey);
	}
	@Override
	public void mouseEntered(MouseEvent arg0) {}
	@Override
	public void mouseExited(MouseEvent arg0) {}
	@Override
	public void mousePressed(MouseEvent arg0) {}
	@Override
	public void mouseReleased(MouseEvent arg0) {}
	
	@SuppressWarnings("serial")
	class ModulePanel extends JPanel {
		private ModulePainter painter;
		public void paintComponent(Graphics g) {
			super.paintComponent(g);
			if (painter != null) painter.modulePaint(g);
		}
		void setModulePainter(ModulePainter p) {
			painter = p;
		}
		@Override
		public Dimension getPreferredSize() {
			int d = (int)Math.round(Math.sqrt(numVars));
			return new Dimension(d * gUnit * 6, d * gUnit);
		}
	}
	class ModulePainter {
		public void modulePaint(Graphics g) {}
	}
	class GridPainter {
		public void paintCell(Graphics g, int c, int r) {
			g.setColor(Color.BLACK);
			g.drawRect(c*gUnit, r*gUnit, gUnit, gUnit);
		}
	}
	
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
