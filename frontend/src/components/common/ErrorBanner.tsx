interface Props {
  message: string;
  onDismiss?: () => void;
}

export default function ErrorBanner({ message, onDismiss }: Props) {
  return (
    <div className="flex items-start gap-3 bg-loss/10 border border-loss/30 rounded-lg px-4 py-3 text-sm text-loss">
      <span className="mt-0.5">✕</span>
      <span className="flex-1">{message}</span>
      {onDismiss && (
        <button onClick={onDismiss} className="text-loss/70 hover:text-loss transition-colors">
          ✕
        </button>
      )}
    </div>
  );
}
