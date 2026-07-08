import { ErrorBanner } from 'freqpred-dashboard'

export function Default() {
  return <ErrorBanner message="Failed to load positions: request timed out." />
}
