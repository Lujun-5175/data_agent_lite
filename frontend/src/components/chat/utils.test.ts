import { describe, expect, it } from 'vitest';
import { buildMessageHistory, normalizeUploadedDataset, parseSseBlocks, parseSseEventBlock } from './utils';

describe('chat utils', () => {
  it('normalizes uploaded dataset payloads', () => {
    const dataset = normalizeUploadedDataset(
      {
        dataset_id: 'abc',
        original_filename: 'demo.csv',
        preview: [{ a: 1 }],
        columns: [{ name: 'a', type: 'numerical' }],
        original_row_count: 3,
        row_count: 3,
        column_count: 1,
        preview_count: 1,
        analysis_basis: 'raw_df',
        preprocessing_log: ['step-1'],
      },
      'fallback.csv'
    );

    expect(dataset.datasetId).toBe('abc');
    expect(dataset.filename).toBe('demo.csv');
    expect(dataset.previewCount).toBe(1);
    expect(dataset.preprocessingLog).toEqual(['step-1']);
  });

  it('builds assistant/user history from text messages only', () => {
    const history = buildMessageHistory([
      { id: '1', type: 'assistant', kind: 'text', content: 'hello', timestamp: new Date() },
      { id: '2', type: 'assistant', kind: 'status', content: 'working', timestamp: new Date() },
      { id: '3', type: 'user', kind: 'text', content: 'world', timestamp: new Date() },
    ]);

    expect(history).toEqual([
      { type: 'ai', content: 'hello' },
      { type: 'human', content: 'world' },
    ]);
  });

  it('parses SSE buffers across chunk boundaries', () => {
    const partial = parseSseBlocks('event: message_chunk\ndata: {"content":"hi"}\n\nevent: done');
    expect(partial.blocks).toHaveLength(1);
    expect(partial.remainder).toBe('event: done');

    const complete = parseSseEventBlock(partial.blocks[0]);
    expect(complete).toEqual({
      eventType: 'message_chunk',
      payload: { content: 'hi' },
    });
  });
});
