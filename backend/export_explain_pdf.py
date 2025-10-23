"""
Eksport raportu Explainability do PDF.

Funkcjonalności:
- Generowanie PDF z ReportLab
- Spis treści
- Numeracja stron
- Wykresy i tabele
- Feature importance
- Metryki i wnioski
"""

import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.error_handler import ExportException, handle_errors
from backend.utils import get_timestamp, sanitize_filename
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ExplainabilityPDFExporter:
    """
    Eksporter raportów Explainability do PDF.
    
    Wykorzystuje ReportLab do tworzenia profesjonalnych PDF-ów.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Inicjalizacja exportera.

        Args:
            output_dir: Katalog wyjściowy (None = outputs/)
        """
        self.output_dir = output_dir or settings.outputs_dir
        self.output_dir.mkdir(exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Konfiguruje niestandardowe style."""
        # Tytuł główny
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=1  # CENTER
        ))

        # Nagłówek sekcji
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12,
            spaceBefore=12
        ))

    @handle_errors(show_in_ui=False)
    def export_report(
        self,
        pipeline_results: Dict,
        feature_importance: Optional[Dict] = None,
        plots: Optional[Dict[str, BytesIO]] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Eksportuje raport do PDF.

        Args:
            pipeline_results: Wyniki pipeline ML
            feature_importance: Feature importance data
            plots: Słownik z wykresami
            filename: Nazwa pliku (None = auto)

        Returns:
            Path: Ścieżka do utworzonego PDF

        Raises:
            ExportException: Gdy eksport się nie powiedzie

        Example:
            >>> exporter = ExplainabilityPDFExporter()
            >>> pipeline = {'problem_type': 'classification', 'n_models_trained': 3}
            >>> # pdf_path = exporter.export_report(pipeline)
        """
        if filename is None:
            timestamp = get_timestamp()
            filename = f"tmiv_explainability_report_{timestamp}.pdf"

        filename = sanitize_filename(filename)
        pdf_path = self.output_dir / filename

        logger.info(f"Tworzenie raportu PDF: {pdf_path}")

        try:
            # Utwórz dokument
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            # Zbuduj treść
            story = []

            # 1. Strona tytułowa
            story.extend(self._create_title_page(pipeline_results))
            story.append(PageBreak())

            # 2. Spis treści
            story.extend(self._create_table_of_contents())
            story.append(PageBreak())

            # 3. Podsumowanie wykonawcze
            story.extend(self._create_executive_summary(pipeline_results))
            story.append(PageBreak())

            # 4. Opis danych
            story.extend(self._create_data_description(pipeline_results))
            story.append(PageBreak())

            # 5. Wyniki modeli
            story.extend(self._create_model_results(pipeline_results))
            story.append(PageBreak())

            # 6. Feature Importance
            if feature_importance:
                story.extend(self._create_feature_importance_section(feature_importance))
                story.append(PageBreak())

            # 7. Wizualizacje
            if plots:
                story.extend(self._create_visualizations_section(plots))
                story.append(PageBreak())

            # 8. Wnioski i rekomendacje
            story.extend(self._create_conclusions(pipeline_results))

            # Zbuduj PDF
            doc.build(story)

            logger.info(f"Raport PDF utworzony: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
            return pdf_path

        except Exception as e:
            raise ExportException(f"Błąd tworzenia raportu PDF: {e}")

    def _create_title_page(self, pipeline_results: Dict) -> List:
        """Tworzy stronę tytułową."""
        elements = []

        # Tytuł
        elements.append(Spacer(1, 2 * inch))
        elements.append(Paragraph(
            "TMIV - The Most Important Variables",
            self.styles['CustomTitle']
        ))

        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(
            "Model Explainability Report",
            self.styles['Heading2']
        ))

        # Informacje
        elements.append(Spacer(1, inch))

        info_data = [
            ["Problem Type:", pipeline_results.get('problem_type', 'N/A')],
            ["Models Trained:", str(pipeline_results.get('n_models_trained', 'N/A'))],
            ["Features:", str(pipeline_results.get('n_features', 'N/A'))],
            ["Best Model:", pipeline_results.get('best_model_name', 'N/A')],
            ["Report Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]

        info_table = Table(info_data, colWidths=[2 * inch, 3 * inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))

        elements.append(info_table)

        return elements

    def _create_table_of_contents(self) -> List:
        """Tworzy spis treści."""
        elements = []

        elements.append(Paragraph("Table of Contents", self.styles['Heading1']))
        elements.append(Spacer(1, 0.3 * inch))

        toc_items = [
            "1. Executive Summary",
            "2. Data Description",
            "3. Model Results",
            "4. Feature Importance",
            "5. Visualizations",
            "6. Conclusions and Recommendations",
        ]

        for item in toc_items:
            elements.append(Paragraph(item, self.styles['Normal']))
            elements.append(Spacer(1, 0.1 * inch))

        return elements

    def _create_executive_summary(self, pipeline_results: Dict) -> List:
        """Tworzy podsumowanie wykonawcze."""
        elements = []

        elements.append(Paragraph("1. Executive Summary", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        summary_text = f"""
        This report presents the results of machine learning model training and evaluation 
        for a {pipeline_results.get('problem_type', 'N/A')} problem. 
        A total of {pipeline_results.get('n_models_trained', 'N/A')} models were trained 
        using {pipeline_results.get('n_features', 'N/A')} features.
        
        The best performing model was {pipeline_results.get('best_model_name', 'N/A')}.
        """

        elements.append(Paragraph(summary_text.strip(), self.styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_data_description(self, pipeline_results: Dict) -> List:
        """Tworzy sekcję opisu danych."""
        elements = []

        elements.append(Paragraph("2. Data Description", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        # Tabela z opisem
        data_info = [
            ["Metric", "Value"],
            ["Number of Features", str(pipeline_results.get('n_features', 'N/A'))],
            ["Problem Type", pipeline_results.get('problem_type', 'N/A')],
        ]

        # Dodaj class names jeśli dostępne
        class_names = pipeline_results.get('class_names')
        if class_names:
            data_info.append(["Classes", ", ".join(map(str, class_names))])

        data_table = Table(data_info, colWidths=[2.5 * inch, 3.5 * inch])
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(data_table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_model_results(self, pipeline_results: Dict) -> List:
        """Tworzy sekcję wyników modeli."""
        elements = []

        elements.append(Paragraph("3. Model Results", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        # Tabela z rankingiem modeli
        ranking = pipeline_results.get('model_ranking', [])

        if ranking:
            table_data = [["Rank", "Model", "Score", "Time (s)"]]

            for idx, model_info in enumerate(ranking[:10], 1):
                table_data.append([
                    str(idx),
                    model_info['model_name'],
                    f"{model_info['score']:.4f}",
                    f"{model_info['training_time']:.2f}"
                ])

            results_table = Table(table_data, colWidths=[0.7 * inch, 2.5 * inch, 1.5 * inch, 1.5 * inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))

            elements.append(results_table)
        else:
            elements.append(Paragraph("No model ranking available.", self.styles['Normal']))

        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_feature_importance_section(self, feature_importance: Dict) -> List:
        """Tworzy sekcję feature importance."""
        elements = []

        elements.append(Paragraph("4. Feature Importance", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(
            "The following features were identified as most important for predictions:",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 0.2 * inch))

        # Przykładowa tabela (w rzeczywistości użyj danych z feature_importance)
        # TODO: Dostosuj do rzeczywistych danych

        return elements

    def _create_visualizations_section(self, plots: Dict[str, BytesIO]) -> List:
        """Tworzy sekcję wizualizacji."""
        elements = []

        elements.append(Paragraph("5. Visualizations", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        for plot_name, plot_bytes in plots.items():
            if plot_bytes is None:
                continue

            try:
                plot_bytes.seek(0)
                img = Image(plot_bytes, width=5 * inch, height=3 * inch)
                elements.append(Paragraph(plot_name, self.styles['Normal']))
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(img)
                elements.append(Spacer(1, 0.3 * inch))
            except Exception as e:
                logger.warning(f"Nie udało się dodać wykresu {plot_name} do PDF: {e}")

        return elements

    def _create_conclusions(self, pipeline_results: Dict) -> List:
        """Tworzy sekcję wniosków."""
        elements = []

        elements.append(Paragraph("6. Conclusions and Recommendations", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2 * inch))

        conclusions_text = f"""
        Based on the analysis, the {pipeline_results.get('best_model_name', 'best model')} 
        demonstrated superior performance for this {pipeline_results.get('problem_type', '')} task.
        
        Key recommendations:
        • Consider deploying the top-performing model for production use
        • Monitor model performance regularly
        • Retrain periodically with new data
        • Ensure proper feature preprocessing in production
        
        Limitations:
        • Model performance may vary on unseen data
        • Results are based on the specific dataset used
        • Feature importance may change with different data distributions
        """

        elements.append(Paragraph(conclusions_text.strip(), self.styles['Normal']))

        return elements


@handle_errors(show_in_ui=False)
def create_sample_pdf() -> Path:
    """
    Tworzy przykładowy PDF (do testów).

    Returns:
        Path: Ścieżka do utworzonego PDF
    """
    exporter = ExplainabilityPDFExporter()

    pipeline_results = {
        'problem_type': 'binary_classification',
        'n_models_trained': 5,
        'n_features': 15,
        'best_model_name': 'RandomForest',
        'model_ranking': [
            {'model_name': 'RandomForest', 'score': 0.92, 'training_time': 12.5},
            {'model_name': 'XGBoost', 'score': 0.91, 'training_time': 8.3},
        ]
    }

    return exporter.export_report(pipeline_results)