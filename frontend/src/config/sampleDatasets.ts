export interface SampleDataset {
  id: string;
  name: string;
  filename: string;
  description: string;
  rowCount: number;
  columnCount: number;
  path: string;
  prompts: string[];
}

export const SAMPLE_DATASETS: SampleDataset[] = [
  {
    id: 'sales',
    name: 'Sales Data',
    filename: 'sales_data.csv',
    description: 'Explore sales trends across regions, categories, customer segments, and channels.',
    rowCount: 800,
    columnCount: 10,
    path: '/sample_data/sales_data.csv',
    prompts: [
      "What's the monthly sales trend? Show me a line chart.",
      'Which product category has the highest revenue? Compare across regions.',
      'Run a t-test comparing total_amount between online and offline channels.',
    ],
  },
  {
    id: 'students',
    name: 'Student Scores',
    filename: 'student_scores.csv',
    description: 'Analyze score drivers, study habits, attendance, and grade outcomes.',
    rowCount: 300,
    columnCount: 9,
    path: '/sample_data/student_scores.csv',
    prompts: [
      "What's the correlation between study_hours and final_score?",
      'Run a linear regression to predict final_score from study_hours and attendance_rate.',
      'Is there a significant difference in scores between male and female students? Run a t-test.',
    ],
  },
  {
    id: 'behavior',
    name: 'User Behavior',
    filename: 'user_behavior.csv',
    description: 'Compare conversion rates, A/B test outcomes, channels, devices, and engagement segments.',
    rowCount: 1000,
    columnCount: 9,
    path: '/sample_data/user_behavior.csv',
    prompts: [
      'Compare conversion_flag rates between ab_group A and B. Run a chi-square test.',
      'Which channel_source has the highest conversion rate?',
      'Segment users by session_count and visualize the distribution.',
    ],
  },
];
