/**
 * Competitive Price Intelligence - Documentation Site
 * Shared JavaScript for all pages
 */

(function () {
    'use strict';

    // =========================================================================
    // 1. Sidebar Toggle (Mobile)
    // =========================================================================
    function initSidebar() {
        var menuBtn = document.querySelector('.mobile-menu-btn');
        var sidebar = document.querySelector('.sidebar');
        if (!menuBtn || !sidebar) return;

        menuBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            sidebar.classList.toggle('open');
        });

        document.addEventListener('click', function (event) {
            if (window.innerWidth <= 1024 &&
                !sidebar.contains(event.target) &&
                !menuBtn.contains(event.target)) {
                sidebar.classList.remove('open');
            }
        });
    }

    // =========================================================================
    // 2. Smooth Scroll for Anchor Links
    // =========================================================================
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
            anchor.addEventListener('click', function (e) {
                var href = this.getAttribute('href');
                if (href === '#') return;
                var target = document.querySelector(href);
                if (target) {
                    e.preventDefault();
                    var offset = 20;
                    var top = target.getBoundingClientRect().top + window.pageYOffset - offset;
                    window.scrollTo({ top: top, behavior: 'smooth' });
                    // Update URL hash without jumping
                    history.pushState(null, null, href);
                }
            });
        });
    }

    // =========================================================================
    // 3. Active Nav Link Tracking on Scroll
    // =========================================================================
    function initScrollSpy() {
        var sections = document.querySelectorAll('section[id]');
        var navLinks = document.querySelectorAll('.sidebar .nav-links a[href^="#"]');
        if (sections.length === 0 || navLinks.length === 0) return;

        var tocLinks = document.querySelectorAll('.toc-list a');

        function update() {
            var scrollPos = window.pageYOffset;
            var current = '';
            sections.forEach(function (section) {
                if (scrollPos >= section.offsetTop - 120) {
                    current = section.getAttribute('id');
                }
            });
            navLinks.forEach(function (link) {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + current) {
                    link.classList.add('active');
                }
            });
            // Also update right-side ToC
            tocLinks.forEach(function (link) {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + current) {
                    link.classList.add('active');
                }
            });
        }

        window.addEventListener('scroll', update, { passive: true });
        update();
    }

    // =========================================================================
    // 4. Copy-to-Clipboard for Code Blocks
    // =========================================================================
    function initCopyButtons() {
        document.querySelectorAll('pre').forEach(function (pre) {
            var btn = document.createElement('button');
            btn.className = 'copy-btn';
            btn.textContent = 'Copy';
            btn.setAttribute('aria-label', 'Copy code to clipboard');
            btn.addEventListener('click', function () {
                var code = pre.querySelector('code');
                var text = code ? code.textContent : pre.textContent;
                navigator.clipboard.writeText(text).then(function () {
                    btn.textContent = 'Copied!';
                    btn.classList.add('copied');
                    setTimeout(function () {
                        btn.textContent = 'Copy';
                        btn.classList.remove('copied');
                    }, 2000);
                });
            });
            pre.appendChild(btn);
        });
    }

    // =========================================================================
    // 5. Right-Side "On This Page" Table of Contents
    // =========================================================================
    function initTableOfContents() {
        var tocContainer = document.querySelector('.toc');
        if (!tocContainer) return;

        var headings = document.querySelectorAll('.content-body section[id] > h2');
        if (headings.length < 2) {
            tocContainer.style.display = 'none';
            return;
        }

        var tocTitle = document.createElement('div');
        tocTitle.className = 'toc-title';
        tocTitle.textContent = 'On this page';
        tocContainer.appendChild(tocTitle);

        var list = document.createElement('ul');
        list.className = 'toc-list';

        headings.forEach(function (heading) {
            var section = heading.closest('section[id]');
            if (!section) return;
            var li = document.createElement('li');
            var a = document.createElement('a');
            a.href = '#' + section.id;
            a.textContent = heading.textContent;
            li.appendChild(a);
            list.appendChild(li);
        });

        tocContainer.appendChild(list);
    }

    // =========================================================================
    // 6. Heading Anchor Links
    // =========================================================================
    function initHeadingAnchors() {
        var headings = document.querySelectorAll('.content-body h2[id], .content-body h3[id], .content-body section[id] > h2, .content-body section[id] > h3');
        headings.forEach(function (heading) {
            var id = heading.id || (heading.closest('section[id]') && heading.closest('section[id]').id);
            if (!id) return;
            // Skip if anchor already exists
            if (heading.querySelector('.heading-anchor')) return;

            var anchor = document.createElement('a');
            anchor.className = 'heading-anchor';
            anchor.href = '#' + id;
            anchor.setAttribute('aria-label', 'Link to this section');
            anchor.textContent = '#';
            heading.style.position = 'relative';
            heading.appendChild(anchor);
        });
    }

    // =========================================================================
    // 7. Client-Side Search (Lunr.js-like simple search)
    // =========================================================================
    var searchIndex = [];

    function buildSearchIndex() {
        var sections = document.querySelectorAll('.content-body section[id]');
        sections.forEach(function (section) {
            var heading = section.querySelector('h2, h3');
            var title = heading ? heading.textContent.replace(/#$/, '').trim() : '';
            var text = section.textContent.substring(0, 500).replace(/\s+/g, ' ').trim();
            searchIndex.push({
                id: section.id,
                title: title,
                text: text
            });
        });
    }

    function performSearch(query) {
        if (!query || query.length < 2) return [];
        var q = query.toLowerCase();
        var results = [];
        searchIndex.forEach(function (entry) {
            var titleMatch = entry.title.toLowerCase().indexOf(q) !== -1;
            var textMatch = entry.text.toLowerCase().indexOf(q) !== -1;
            if (titleMatch || textMatch) {
                results.push({
                    id: entry.id,
                    title: entry.title,
                    titleMatch: titleMatch,
                    snippet: getSnippet(entry.text, q)
                });
            }
        });
        // Sort: title matches first
        results.sort(function (a, b) {
            if (a.titleMatch && !b.titleMatch) return -1;
            if (!a.titleMatch && b.titleMatch) return 1;
            return 0;
        });
        return results.slice(0, 8);
    }

    function getSnippet(text, query) {
        var idx = text.toLowerCase().indexOf(query);
        if (idx === -1) return text.substring(0, 120) + '...';
        var start = Math.max(0, idx - 40);
        var end = Math.min(text.length, idx + query.length + 80);
        var snippet = (start > 0 ? '...' : '') + text.substring(start, end) + (end < text.length ? '...' : '');
        return snippet;
    }

    function highlightMatch(text, query) {
        if (!query) return text;
        var regex = new RegExp('(' + query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }

    function initSearch() {
        var searchContainer = document.querySelector('.search-container');
        if (!searchContainer) return;

        var input = searchContainer.querySelector('.search-input');
        var resultsPanel = searchContainer.querySelector('.search-results');
        var clearBtn = searchContainer.querySelector('.search-clear');
        if (!input || !resultsPanel) return;

        buildSearchIndex();

        input.addEventListener('input', function () {
            var query = this.value.trim();
            if (clearBtn) {
                clearBtn.style.display = query.length > 0 ? 'block' : 'none';
            }
            if (query.length < 2) {
                resultsPanel.style.display = 'none';
                resultsPanel.innerHTML = '';
                return;
            }
            var results = performSearch(query);
            if (results.length === 0) {
                resultsPanel.innerHTML = '<div class="search-no-results">No results found</div>';
                resultsPanel.style.display = 'block';
                return;
            }
            var html = results.map(function (r) {
                return '<a class="search-result-item" href="#' + r.id + '">' +
                    '<div class="search-result-title">' + highlightMatch(escapeHtml(r.title), query) + '</div>' +
                    '<div class="search-result-snippet">' + highlightMatch(escapeHtml(r.snippet), query) + '</div>' +
                    '</a>';
            }).join('');
            resultsPanel.innerHTML = html;
            resultsPanel.style.display = 'block';
        });

        if (clearBtn) {
            clearBtn.addEventListener('click', function () {
                input.value = '';
                resultsPanel.style.display = 'none';
                resultsPanel.innerHTML = '';
                clearBtn.style.display = 'none';
                input.focus();
            });
        }

        // Close results when clicking outside
        document.addEventListener('click', function (e) {
            if (!searchContainer.contains(e.target)) {
                resultsPanel.style.display = 'none';
            }
        });

        // Close results and navigate when clicking a result
        resultsPanel.addEventListener('click', function (e) {
            var item = e.target.closest('.search-result-item');
            if (item) {
                resultsPanel.style.display = 'none';
                input.value = '';
                if (clearBtn) clearBtn.style.display = 'none';
            }
        });

        // Keyboard shortcut: Ctrl/Cmd + K to focus search
        document.addEventListener('keydown', function (e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                input.focus();
                input.select();
            }
            if (e.key === 'Escape') {
                resultsPanel.style.display = 'none';
                input.blur();
            }
        });
    }

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    // =========================================================================
    // 8. Mermaid.js Initialization
    // =========================================================================
    function initMermaid() {
        if (typeof mermaid !== 'undefined') {
            mermaid.initialize({
                startOnLoad: true,
                theme: 'default',
                flowchart: {
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis'
                },
                themeVariables: {
                    primaryColor: '#dbeafe',
                    primaryTextColor: '#1e40af',
                    primaryBorderColor: '#2563eb',
                    lineColor: '#6b7280',
                    secondaryColor: '#f0fdf4',
                    tertiaryColor: '#fef3c7',
                    fontSize: '14px'
                }
            });
        }
    }

    // =========================================================================
    // Initialize Everything on DOMContentLoaded
    // =========================================================================
    document.addEventListener('DOMContentLoaded', function () {
        initSidebar();
        initSmoothScroll();
        initCopyButtons();
        initTableOfContents();
        initScrollSpy();
        initHeadingAnchors();
        initSearch();
        initMermaid();
    });
})();