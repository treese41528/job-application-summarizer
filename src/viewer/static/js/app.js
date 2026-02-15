/**
 * Job Application Summarizer - Client-side JavaScript
 *
 * Handles: live search, keyboard shortcuts, collapsible sections.
 */

document.addEventListener('DOMContentLoaded', () => {

    // ── Keyboard Shortcuts ──
    document.addEventListener('keydown', (e) => {
        // '/' to focus search
        if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
            const searchBox = document.getElementById('search-box');
            if (searchBox && document.activeElement !== searchBox) {
                e.preventDefault();
                searchBox.focus();
            }
        }

        // Escape to blur search
        if (e.key === 'Escape') {
            document.activeElement.blur();
        }
    });

    // ── Collapsible Sections ──
    document.querySelectorAll('[data-collapse]').forEach(trigger => {
        trigger.addEventListener('click', () => {
            const targetId = trigger.getAttribute('data-collapse');
            const target = document.getElementById(targetId);
            if (target) {
                target.classList.toggle('hidden');
                const icon = trigger.querySelector('.collapse-icon');
                if (icon) {
                    icon.textContent = target.classList.contains('hidden') ? '▸' : '▾';
                }
            }
        });
    });

    // ── Smooth scroll for anchor links ──
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(anchor.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    console.log('VAP Application Viewer loaded. Press "/" to search.');
});
