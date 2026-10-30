"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ViewportErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Three.js Viewport Error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "#a1a1aa",
          backgroundColor: "#0d0d0f",
          padding: "24px",
          textAlign: "center"
        }}>
          <h3 style={{ color: "#ef4444", marginBottom: "8px", fontSize: "14px" }}>
            WebGL Rendering Encountered an Issue
          </h3>
          <p style={{ fontSize: "12px", maxWidth: "400px", lineHeight: "1.5" }}>
            {this.state.error?.message || "An unexpected error occurred inside the 3D viewport canvas."}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginTop: "16px",
              padding: "6px 14px",
              backgroundColor: "#27272a",
              color: "#f4f4f5",
              border: "1px solid #3f3f46",
              borderRadius: "4px",
              fontSize: "12px",
              cursor: "pointer"
            }}
          >
            Retry Viewport
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

