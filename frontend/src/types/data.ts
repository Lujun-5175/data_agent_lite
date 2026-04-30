export type ColumnType = 'numerical' | 'categorical';

export interface UploadedColumn {
  name: string;
  type: ColumnType;
}

export interface UploadedDataset {
  datasetId: string;
  filename: string;
  preview: Array<Record<string, unknown>>;
  columns: UploadedColumn[];
  originalRowCount: number;
  rowCount: number;
  columnCount: number;
  previewCount: number;
  analysisBasis: 'working_df' | string;
  preprocessingLog: string[];
}

export interface ServerUploadResponse {
  status?: string;
  message?: string;
  dataset_id?: string;
  original_filename?: string;
  filename?: string;
  preview?: Array<Record<string, unknown>>;
  columns?: UploadedColumn[];
  original_row_count?: number;
  row_count?: number;
  column_count?: number;
  preview_count?: number;
  analysis_basis?: string;
  preprocessing_log?: string[];
}
