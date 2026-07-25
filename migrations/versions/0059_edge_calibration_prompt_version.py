"""Add prompt_version dimension to edge_calibration_scores.

The assessor's calibration block was pooling every signal prompt version ever
run. That describes a model production no longer resembles: on KXTRUMPSAY the
NO-side profit edge measures -0.240 under signal-v7, -0.067 under v4, +0.120
under v9 and +0.133 under v11 (current). The pooled all-time figure is +0.073,
roughly half the current regime's and contaminated by dead eras.

NULL keeps its existing meaning — the all-versions rollup — so every historical
row stays valid and serves as the fallback when a version cohort is too thin.
Non-NULL rows are the new version-scoped cells.

Revision ID: 0059
Revises: 0058
"""
import sqlalchemy as sa
from alembic import op

revision = "0059"
down_revision = "0058"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "edge_calibration_scores",
        sa.Column("prompt_version", sa.VARCHAR(100), nullable=True),
    )
    # The loader looks up (band, direction, series, prompt_version) at the latest
    # computed_at; without prompt_version in the index that lookup degrades to a
    # scan of every version's rows for the band.
    op.drop_index(
        "ix_edge_calibration_scores_band_direction_series_computed",
        table_name="edge_calibration_scores",
    )
    op.create_index(
        "ix_edge_calib_band_dir_series_version_computed",
        "edge_calibration_scores",
        ["edge_band", "direction", "series_ticker", "prompt_version", "computed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_edge_calib_band_dir_series_version_computed",
        table_name="edge_calibration_scores",
    )
    op.create_index(
        "ix_edge_calibration_scores_band_direction_series_computed",
        "edge_calibration_scores",
        ["edge_band", "direction", "series_ticker", "computed_at"],
    )
    # Version-scoped rows are meaningless without the column; drop them so a
    # downgraded loader does not double-count them against the NULL rollups.
    op.execute("DELETE FROM edge_calibration_scores WHERE prompt_version IS NOT NULL")
    op.drop_column("edge_calibration_scores", "prompt_version")
