interface Props {
  title: string;
  subtitle?: string;
}

export default function Navbar({ title, subtitle }: Props) {
  return (
    <div className="flex items-center justify-between py-5 mb-6 border-b border-dim">
      <div>
        <h1 className="text-prose text-xl font-bold">{title}</h1>
        {subtitle && <p className="text-muted text-sm mt-0.5">{subtitle}</p>}
      </div>
      <a
        href="https://github.com"
        target="_blank"
        rel="noreferrer"
        className="text-muted hover:text-prose transition-colors text-sm font-medium"
      >
        GitHub ↗
      </a>
    </div>
  );
}
