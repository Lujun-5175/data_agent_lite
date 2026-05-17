import { describe, expect, it } from 'vitest';
import {
  buildMessageHistory,
  normalizeRouteInfo,
  normalizeUploadedDataset,
  parseSseBlocks,
  parseSseEventBlock,
  summarizeRouteInfo,
} from './utils';

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

  it('normalizes route info payloads', () => {
    const routeInfo = normalizeRouteInfo({
      primary_mode: 'dataset_overview',
      confidence_score: 0.72,
      intent_type: 'dataset_overview',
      confidence: 'medium',
      route_source: 'llm_primary',
      conflict_flags: ['dataset_overview_missed'],
      ambiguity_flags: ['dataset_overview_missed'],
      guardrail_actions: [],
      fallback_reasons: [],
      suggested_plan: ['inspect schema', 'summarize dataset'],
      requested_capabilities: ['summarize_dataset', 'inspect_schema'],
      requires_ml: false,
      requires_chart: false,
      requires_python_analysis: true,
      is_follow_up: false,
      needs_dataset: true,
      needs_tool_execution: false,
      needs_artifact_context: false,
      final_branch: 'dataset_overview',
    });

    expect(routeInfo).not.toBeNull();
    expect(routeInfo?.primaryMode).toBe('dataset_overview');
    expect(routeInfo?.confidenceScore).toBe(0.72);
    expect(routeInfo?.requestedCapabilities).toEqual(['summarize_dataset', 'inspect_schema']);
    expect(routeInfo?.finalBranch).toBe('dataset_overview');
    expect(routeInfo?.intentType).toBe('dataset_overview');
    expect(summarizeRouteInfo(routeInfo!)).toContain('数据概览');
  });
});
